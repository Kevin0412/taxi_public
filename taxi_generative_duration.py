import sys
import os
import math
import json
import random
import argparse
from dataclasses import dataclass

local_site = f"/home/test/.local/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
if os.path.isdir(local_site) and local_site not in sys.path:
    sys.path.insert(0, local_site)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from taxi_duration_prediction import (
    load_real_data,
    haversine,
    evaluate_predictions,
    mape,
    OUTPUT_DIR,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class Sample:
    points: list
    hour: int
    weekday: int
    prefix_points: int


class Standardizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, arrays):
        x = np.concatenate(arrays, axis=0)
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = x.std(axis=0, keepdims=True) + 1e-6
        return self

    def transform(self, x):
        return (x - self.mean) / self.std


class TrajectorySequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


class CausalTrajectoryTransformer(nn.Module):
    def __init__(self, token_dim, d_model=96, nhead=4, num_layers=3, dropout=0.12, max_len=128):
        super().__init__()
        self.token_proj = nn.Sequential(
            nn.Linear(token_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.delta_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )
        self.stop_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, tokens, padding_mask=None):
        seq_len = tokens.size(1)
        if seq_len > self.pos_embed.size(1):
            repeat_len = seq_len - self.pos_embed.size(1)
            extra = self.pos_embed[:, -1:, :].repeat(1, repeat_len, 1)
            pos_embed = torch.cat([self.pos_embed, extra], dim=1)
        else:
            pos_embed = self.pos_embed
        x = self.token_proj(tokens) + pos_embed[:, :seq_len, :]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        x = self.norm(x)
        delta = self.delta_head(x)
        stop_logit = self.stop_head(x).squeeze(-1)
        return delta, stop_logit


class LossComputer:
    def __init__(self, delta_weight=10.0, stop_weight=1.0, pos_weight=5.0):
        self.delta_weight = delta_weight
        self.stop_weight = stop_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([pos_weight], dtype=torch.float32))

    def __call__(self, pred_delta, pred_stop, target_delta, target_stop, loss_mask):
        delta_l1 = torch.abs(pred_delta - target_delta).sum(dim=-1)
        delta_loss = (delta_l1 * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)

        stop_loss_raw = self.bce(pred_stop, target_stop)
        stop_loss = (stop_loss_raw * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)

        total = self.delta_weight * delta_loss + self.stop_weight * stop_loss
        return total, delta_loss.detach(), stop_loss.detach()


FOURIER_SCALES = [0.01, 0.02]


def build_fourier_features(value):
    feats = []
    for scale in FOURIER_SCALES:
        angle = 2.0 * math.pi * value / scale
        feats.extend([math.sin(angle), math.cos(angle)])
    return feats


def point_token_features(points, idx, hour, weekday):
    start_lon, start_lat = points[0]
    curr_lon, curr_lat = points[idx]
    rel_dx = curr_lon - start_lon
    rel_dy = curr_lat - start_lat

    if idx == 0:
        delta_x = 0.0
        delta_y = 0.0
        seg_dist = 0.0
        speed = 0.0
    else:
        prev_lon, prev_lat = points[idx - 1]
        delta_x = curr_lon - prev_lon
        delta_y = curr_lat - prev_lat
        seg_dist = haversine(prev_lat, prev_lon, curr_lat, curr_lon)
        speed = seg_dist / 15.0

    hour_angle = 2.0 * math.pi * hour / 24.0
    weekday_angle = 2.0 * math.pi * weekday / 7.0

    feats = [
        rel_dx,
        rel_dy,
        delta_x,
        delta_y,
        seg_dist / 100.0,
        speed / 15.0,
        math.sin(hour_angle),
        math.cos(hour_angle),
        math.sin(weekday_angle),
        math.cos(weekday_angle),
    ]
    feats.extend(build_fourier_features(rel_dx))
    feats.extend(build_fourier_features(rel_dy))
    return feats


def build_sample_from_row(row, prefix_points):
    points = row['_poly_clean']
    if len(points) <= prefix_points:
        return None
    ts = int(row['TIMESTAMP'])
    hour = (ts // 3600) % 24
    weekday = (ts // 86400) % 7
    return Sample(points=points, hour=hour, weekday=weekday, prefix_points=prefix_points)


def prepare_samples(df, prefix_points=20, max_samples=9000, max_total_points=96, min_future_points=3, seed=42):
    samples = []
    for _, row in df.iterrows():
        sample = build_sample_from_row(row, prefix_points=prefix_points)
        if sample is None:
            continue
        if len(sample.points) > max_total_points:
            sample = Sample(
                points=sample.points[:max_total_points],
                hour=sample.hour,
                weekday=sample.weekday,
                prefix_points=sample.prefix_points,
            )
        future_points = len(sample.points) - prefix_points
        if future_points < min_future_points:
            continue
        samples.append(sample)

    rng = np.random.default_rng(seed)
    if len(samples) > max_samples:
        idx = rng.choice(len(samples), size=max_samples, replace=False)
        samples = [samples[i] for i in idx]
    return samples


def sample_to_training_arrays(sample):
    points = sample.points
    hour = sample.hour
    weekday = sample.weekday
    prefix_points = sample.prefix_points
    input_len = len(points) - 1

    tokens = np.asarray(
        [point_token_features(points, i, hour, weekday) for i in range(input_len)],
        dtype=np.float32,
    )
    target_delta = []
    target_stop = []
    loss_mask = []
    for i in range(input_len):
        curr_lon, curr_lat = points[i]
        next_lon, next_lat = points[i + 1]
        target_delta.append([next_lon - curr_lon, next_lat - curr_lat])
        target_stop.append(1.0 if i == input_len - 1 else 0.0)
        loss_mask.append(1.0 if i >= prefix_points - 1 else 0.0)

    meta = {
        'observed_duration': (prefix_points - 1) * 15,
        'true_duration': (len(points) - 1) * 15,
        'true_remaining_steps': len(points) - prefix_points,
        'prefix_points': prefix_points,
        'full_points': len(points),
        'hour': hour,
        'weekday': weekday,
        'start_point': points[0],
        'prefix_last_point': points[prefix_points - 1],
        'true_points': points,
    }

    return {
        'tokens': tokens,
        'target_delta': np.asarray(target_delta, dtype=np.float32),
        'target_stop': np.asarray(target_stop, dtype=np.float32),
        'loss_mask': np.asarray(loss_mask, dtype=np.float32),
        'meta': meta,
    }


def build_sequences(samples):
    return [sample_to_training_arrays(sample) for sample in samples]


def collate_batch(batch):
    max_len = max(item['tokens'].shape[0] for item in batch)
    token_dim = batch[0]['tokens'].shape[1]
    bs = len(batch)

    tokens = np.zeros((bs, max_len, token_dim), dtype=np.float32)
    target_delta = np.zeros((bs, max_len, 2), dtype=np.float32)
    target_stop = np.zeros((bs, max_len), dtype=np.float32)
    loss_mask = np.zeros((bs, max_len), dtype=np.float32)
    padding_mask = np.ones((bs, max_len), dtype=bool)
    metas = []

    for i, item in enumerate(batch):
        length = item['tokens'].shape[0]
        tokens[i, :length] = item['tokens']
        target_delta[i, :length] = item['target_delta']
        target_stop[i, :length] = item['target_stop']
        loss_mask[i, :length] = item['loss_mask']
        padding_mask[i, :length] = False
        metas.append(item['meta'])

    return {
        'tokens': torch.tensor(tokens, dtype=torch.float32),
        'target_delta': torch.tensor(target_delta, dtype=torch.float32),
        'target_stop': torch.tensor(target_stop, dtype=torch.float32),
        'loss_mask': torch.tensor(loss_mask, dtype=torch.float32),
        'padding_mask': torch.tensor(padding_mask, dtype=torch.bool),
        'metas': metas,
    }


def evaluate_loader(model, loader, loss_computer, device):
    model.eval()
    total_loss = 0.0
    total_delta = 0.0
    total_stop = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            tokens = batch['tokens'].to(device)
            target_delta = batch['target_delta'].to(device)
            target_stop = batch['target_stop'].to(device)
            loss_mask = batch['loss_mask'].to(device)
            padding_mask = batch['padding_mask'].to(device)

            pred_delta, pred_stop = model(tokens, padding_mask=padding_mask)
            loss, delta_loss, stop_loss = loss_computer(
                pred_delta,
                pred_stop,
                target_delta,
                target_stop,
                loss_mask,
            )
            total_loss += float(loss.item())
            total_delta += float(delta_loss.item())
            total_stop += float(stop_loss.item())
            n_batches += 1
    return {
        'loss': total_loss / max(n_batches, 1),
        'delta_loss': total_delta / max(n_batches, 1),
        'stop_loss': total_stop / max(n_batches, 1),
    }


def train_model(train_ds, val_ds, token_dim, max_len, args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"训练设备: {device}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    model = CausalTrajectoryTransformer(
        token_dim=token_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dropout=args.dropout,
        max_len=max_len,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_computer = LossComputer(
        delta_weight=args.delta_weight,
        stop_weight=args.stop_weight,
        pos_weight=args.stop_pos_weight,
    )
    loss_computer.bce = loss_computer.bce.to(device)

    best_state = None
    best_val = float('inf')
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_delta = 0.0
        train_stop = 0.0
        n_batches = 0

        for batch in train_loader:
            tokens = batch['tokens'].to(device)
            target_delta = batch['target_delta'].to(device)
            target_stop = batch['target_stop'].to(device)
            loss_mask = batch['loss_mask'].to(device)
            padding_mask = batch['padding_mask'].to(device)

            optimizer.zero_grad()
            pred_delta, pred_stop = model(tokens, padding_mask=padding_mask)
            loss, delta_loss, stop_loss = loss_computer(
                pred_delta,
                pred_stop,
                target_delta,
                target_stop,
                loss_mask,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += float(loss.item())
            train_delta += float(delta_loss.item())
            train_stop += float(stop_loss.item())
            n_batches += 1

        train_stats = {
            'loss': train_loss / max(n_batches, 1),
            'delta_loss': train_delta / max(n_batches, 1),
            'stop_loss': train_stop / max(n_batches, 1),
        }
        val_stats = evaluate_loader(model, val_loader, loss_computer, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_stats['loss']:.4f} "
            f"(delta {train_stats['delta_loss']:.4f}, stop {train_stats['stop_loss']:.4f}) | "
            f"val loss {val_stats['loss']:.4f} "
            f"(delta {val_stats['delta_loss']:.4f}, stop {val_stats['stop_loss']:.4f})"
        )

        if val_stats['loss'] < best_val - 1e-4:
            best_val = val_stats['loss']
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, device


def rollout_one(model, sample_meta, standardizer, device, stop_threshold=0.5, max_gen_steps=80, temperature=0.0):
    use_stochastic = temperature > 0
    model.train(mode=use_stochastic)

    points = [list(pt) for pt in sample_meta['true_points'][:sample_meta['prefix_points']]]
    hour = sample_meta['hour']
    weekday = sample_meta['weekday']
    generated_steps = 0
    stop_probs = []

    for _ in range(max_gen_steps):
        token_arr = np.asarray(
            [point_token_features(points, i, hour, weekday) for i in range(len(points))],
            dtype=np.float32,
        )
        token_arr = standardizer.transform(token_arr)
        tokens = torch.tensor(token_arr[None, :, :], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred_delta, pred_stop = model(tokens, padding_mask=None)
            delta = pred_delta[0, -1].detach().cpu().numpy()
            stop_logit = pred_stop[0, -1].item()
            if use_stochastic:
                stop_prob = torch.sigmoid(torch.tensor(stop_logit / max(temperature, 1e-6))).item()
                stop_decision = np.random.rand() < stop_prob
            else:
                stop_prob = torch.sigmoid(torch.tensor(stop_logit)).item()
                stop_decision = stop_prob >= stop_threshold

        last_lon, last_lat = points[-1]
        next_lon = float(last_lon + delta[0])
        next_lat = float(last_lat + delta[1])
        points.append([next_lon, next_lat])
        generated_steps += 1
        stop_probs.append(stop_prob)
        if stop_decision:
            break

    pred_duration = sample_meta['observed_duration'] + generated_steps * 15
    return {
        'pred_duration': pred_duration,
        'generated_steps': generated_steps,
        'stop_probs': stop_probs,
        'generated_points': points,
        'stopped': bool(stop_probs and generated_steps < max_gen_steps),
    }


def run_rollout_eval(model, test_sequences, standardizer, device, args, temperature=0.0, num_rollouts=1, label='greedy'):
    y_true = []
    y_pred = []
    true_remaining = []
    pred_remaining = []
    stop_biases = []
    stop_triggered = 0.0

    for item in test_sequences:
        rollouts = [
            rollout_one(
                model,
                item['meta'],
                standardizer,
                device,
                stop_threshold=args.stop_threshold,
                max_gen_steps=args.max_gen_steps,
                temperature=temperature,
            )
            for _ in range(num_rollouts)
        ]

        true_duration = item['meta']['true_duration']
        tr = item['meta']['true_remaining_steps']
        pr = float(np.mean([r['generated_steps'] for r in rollouts]))
        pdur = float(np.mean([r['pred_duration'] for r in rollouts]))
        pstopped = float(np.mean([1.0 if r['stopped'] else 0.0 for r in rollouts]))

        y_true.append(true_duration)
        y_pred.append(pdur)
        true_remaining.append(tr)
        pred_remaining.append(pr)
        stop_biases.append(pr - tr)
        stop_triggered += pstopped

    duration_stats = evaluate_predictions(np.asarray(y_true), np.asarray(y_pred))
    remaining_mae = mean_absolute_error(true_remaining, pred_remaining)
    remaining_rmse = mean_squared_error(true_remaining, pred_remaining) ** 0.5

    return {
        'label': label,
        'temperature': temperature,
        'num_rollouts': num_rollouts,
        'duration_mae': duration_stats['MAE'],
        'duration_rmse': duration_stats['RMSE'],
        'duration_mape': duration_stats['MAPE'],
        'avg_generated_steps': float(np.mean(pred_remaining)),
        'avg_true_remaining_steps': float(np.mean(true_remaining)),
        'remaining_steps_mae': float(remaining_mae),
        'remaining_steps_rmse': float(remaining_rmse),
        'mean_stop_step_bias': float(np.mean(stop_biases)),
        'mean_abs_stop_step_bias': float(np.mean(np.abs(stop_biases))),
        'stop_trigger_rate': float(stop_triggered / max(len(test_sequences), 1)),
        'n_test': len(test_sequences),
    }


def write_summary(args, clean_stats, counts, metrics_list, output_path):
    lines = [
        "# Taxi Generative Duration Prototype",
        "",
        "## Setup",
        f"- data: `{args.data}`",
        f"- raw rows loaded: {clean_stats.get('raw_rows', 'n/a')}",
        f"- cleaned rows kept: {clean_stats.get('after_clean', 'n/a')}",
        f"- sampled usable trips: {counts['all']}",
        f"- split: train={counts['train']}, val={counts['val']}, test={counts['test']}",
        f"- prefix_points: {args.prefix_points}",
        f"- max_total_points: {args.max_total_points}",
        f"- max_gen_steps: {args.max_gen_steps}",
        "",
        "## Model",
        "- causal Transformer with teacher forcing",
        "- per-step outputs: delta_x, delta_y, stop logit",
        "- duration = observed prefix duration + generated_steps × 15s",
        "",
        "## Evaluation",
    ]
    for metrics in metrics_list:
        lines.extend([
            f"### {metrics['label']}",
            f"- temperature: {metrics['temperature']}",
            f"- num_rollouts: {metrics['num_rollouts']}",
            f"- duration MAE: {metrics['duration_mae']:.2f} s",
            f"- duration RMSE: {metrics['duration_rmse']:.2f} s",
            f"- duration MAPE: {metrics['duration_mape']:.2f}%",
            f"- avg generated remaining steps: {metrics['avg_generated_steps']:.2f}",
            f"- avg true remaining steps: {metrics['avg_true_remaining_steps']:.2f}",
            f"- remaining steps MAE: {metrics['remaining_steps_mae']:.2f}",
            f"- remaining steps RMSE: {metrics['remaining_steps_rmse']:.2f}",
            f"- mean stop-step bias: {metrics['mean_stop_step_bias']:.2f}",
            f"- mean abs stop-step bias: {metrics['mean_abs_stop_step_bias']:.2f}",
            f"- stop trigger rate: {metrics['stop_trigger_rate']:.3f}",
            "",
        ])
    lines.extend([
        "## Command",
        "```bash",
        args.run_command,
        "```",
    ])
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='kaggle_dl/train.csv')
    parser.add_argument('--n-sample', type=int, default=12000)
    parser.add_argument('--max-samples', type=int, default=9000)
    parser.add_argument('--prefix-points', type=int, default=20)
    parser.add_argument('--max-total-points', type=int, default=96)
    parser.add_argument('--min-future-points', type=int, default=3)
    parser.add_argument('--test-size', type=float, default=0.15)
    parser.add_argument('--val-size', type=float, default=0.15)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--d-model', type=int, default=96)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.12)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--delta-weight', type=float, default=10.0)
    parser.add_argument('--stop-weight', type=float, default=1.0)
    parser.add_argument('--stop-pos-weight', type=float, default=5.0)
    parser.add_argument('--stop-threshold', type=float, default=0.5)
    parser.add_argument('--max-gen-steps', type=int, default=80)
    parser.add_argument('--stochastic-temperature', type=float, default=2.0)
    parser.add_argument('--stochastic-rollouts', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    args.run_command = "python taxi_generative_duration.py " + " ".join(sys.argv[1:])
    set_seed(args.seed)

    data_path = args.data
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), args.data)

    print("=" * 60)
    print("Taxi generative duration prototype")
    print("=" * 60)
    print(f"Loading real data from: {data_path}")
    df, clean_stats = load_real_data(data_path, n_sample=args.n_sample)

    print("Building trajectory samples...")
    samples = prepare_samples(
        df,
        prefix_points=args.prefix_points,
        max_samples=args.max_samples,
        max_total_points=args.max_total_points,
        min_future_points=args.min_future_points,
        seed=args.seed,
    )
    if len(samples) < 300:
        raise RuntimeError(f"Usable samples too few: {len(samples)}")

    train_samples, test_samples = train_test_split(samples, test_size=args.test_size, random_state=args.seed)
    train_samples, val_samples = train_test_split(train_samples, test_size=args.val_size, random_state=args.seed)

    train_sequences = build_sequences(train_samples)
    val_sequences = build_sequences(val_samples)
    test_sequences = build_sequences(test_samples)

    token_standardizer = Standardizer().fit([item['tokens'] for item in train_sequences])
    for seq_list in [train_sequences, val_sequences, test_sequences]:
        for item in seq_list:
            item['tokens'] = token_standardizer.transform(item['tokens'])

    token_dim = train_sequences[0]['tokens'].shape[1]
    max_len = max(item['tokens'].shape[0] for item in train_sequences + val_sequences + test_sequences)

    counts = {
        'all': len(samples),
        'train': len(train_sequences),
        'val': len(val_sequences),
        'test': len(test_sequences),
    }
    print(
        f"Usable samples: total={counts['all']}, train={counts['train']}, "
        f"val={counts['val']}, test={counts['test']}, token_dim={token_dim}, max_len={max_len}"
    )

    train_ds = TrajectorySequenceDataset(train_sequences)
    val_ds = TrajectorySequenceDataset(val_sequences)
    model, device = train_model(train_ds, val_ds, token_dim, max_len, args)

    print("Running autoregressive rollout on test set...")
    metrics_list = []
    greedy_metrics = run_rollout_eval(
        model, test_sequences, token_standardizer, device, args,
        temperature=0.0, num_rollouts=1, label='greedy_t0'
    )
    metrics_list.append(greedy_metrics)
    if args.stochastic_temperature > 0 and args.stochastic_rollouts > 1:
        stochastic_metrics = run_rollout_eval(
            model, test_sequences, token_standardizer, device, args,
            temperature=args.stochastic_temperature,
            num_rollouts=args.stochastic_rollouts,
            label=f'stochastic_t{args.stochastic_temperature}_mean{args.stochastic_rollouts}'
        )
        metrics_list.append(stochastic_metrics)

    print("\nRollout metrics")
    for metrics in metrics_list:
        print(f"[{metrics['label']}]")
        print(f"  duration MAE   : {metrics['duration_mae']:.2f} s")
        print(f"  duration RMSE  : {metrics['duration_rmse']:.2f} s")
        print(f"  duration MAPE  : {metrics['duration_mape']:.2f}%")
        print(f"  avg gen steps  : {metrics['avg_generated_steps']:.2f}")
        print(f"  avg true steps : {metrics['avg_true_remaining_steps']:.2f}")
        print(f"  stop bias mean : {metrics['mean_stop_step_bias']:.2f}")
        print(f"  stop |bias|    : {metrics['mean_abs_stop_step_bias']:.2f}")
        print(f"  stop triggered : {metrics['stop_trigger_rate']:.3f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUTPUT_DIR, 'taxi_generative_summary.md')
    write_summary(args, clean_stats, counts, metrics_list, summary_path)
    print(f"Summary written to: {summary_path}")


if __name__ == '__main__':
    main()
