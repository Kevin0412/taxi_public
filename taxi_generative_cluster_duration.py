import sys
import os
import math
import random
import argparse
from dataclasses import dataclass

local_site = f"/home/test/.local/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
if os.path.isdir(local_site) and local_site not in sys.path:
    sys.path.insert(0, local_site)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from taxi_duration_prediction import (
    load_real_data,
    evaluate_predictions,
    OUTPUT_DIR,
    PORTO_BBOX,
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

    def inverse_transform(self, x):
        return x * self.std + self.mean


class TrajectoryDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


class ClusterTokenizer:
    def __init__(self, n_clusters=256, batch_size=4096, random_state=42, max_fit_points=200000):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.random_state = random_state
        self.max_fit_points = max_fit_points
        self.kmeans = None
        self.centers = None

    def fit(self, samples):
        pts = []
        for s in samples:
            pts.extend(s.points)
        pts = np.asarray(pts, dtype=np.float32)
        if len(pts) > self.max_fit_points:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(pts), size=self.max_fit_points, replace=False)
            pts = pts[idx]
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=self.batch_size,
            random_state=self.random_state,
            n_init=5,
        )
        self.kmeans.fit(pts)
        self.centers = self.kmeans.cluster_centers_.astype(np.float32)
        return self

    def encode_point(self, lon, lat):
        point = np.asarray([lon, lat], dtype=np.float32)
        d2 = np.sum((self.centers - point[None, :]) ** 2, axis=1)
        cid = int(np.argmin(d2))
        center = self.centers[cid]
        residual = point - center
        return cid, float(residual[0]), float(residual[1])

    def decode_point(self, cid, residual_x, residual_y):
        center = self.centers[int(cid)]
        lon = float(center[0] + residual_x)
        lat = float(center[1] + residual_y)
        lon = float(np.clip(lon, PORTO_BBOX['lon_min'], PORTO_BBOX['lon_max']))
        lat = float(np.clip(lat, PORTO_BBOX['lat_min'], PORTO_BBOX['lat_max']))
        return lon, lat


class ClusterGenerativeTransformer(nn.Module):
    def __init__(self, vocab_size, aux_dim, d_model=128, nhead=4, num_layers=3, dropout=0.1, max_len=128):
        super().__init__()
        self.cluster_embed = nn.Embedding(vocab_size, d_model)
        self.aux_proj = nn.Sequential(
            nn.Linear(aux_dim, d_model),
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
        self.cluster_head = nn.Linear(d_model, vocab_size)
        self.residual_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )
        self.stop_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, cluster_ids, aux_features, padding_mask=None):
        seq_len = cluster_ids.size(1)
        if seq_len > self.pos_embed.size(1):
            extra = self.pos_embed[:, -1:, :].repeat(1, seq_len - self.pos_embed.size(1), 1)
            pos_embed = torch.cat([self.pos_embed, extra], dim=1)
        else:
            pos_embed = self.pos_embed
        x = self.cluster_embed(cluster_ids) + self.aux_proj(aux_features) + pos_embed[:, :seq_len, :]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=cluster_ids.device, dtype=torch.bool), diagonal=1)
        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        x = self.norm(x)
        return self.cluster_head(x), self.residual_head(x), self.stop_head(x).squeeze(-1)


class LossComputer:
    def __init__(self, cluster_weight=1.0, residual_weight=2.0, stop_weight=1.0, stop_pos_weight=4.0):
        self.cluster_weight = cluster_weight
        self.residual_weight = residual_weight
        self.stop_weight = stop_weight
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([stop_pos_weight], dtype=torch.float32))

    def __call__(self, pred_cluster, pred_residual, pred_stop, target_cluster, target_residual, target_stop, loss_mask):
        bs, seq_len, vocab = pred_cluster.shape
        cluster_loss = self.ce(pred_cluster.reshape(bs * seq_len, vocab), target_cluster.reshape(bs * seq_len)).reshape(bs, seq_len)
        cluster_loss = (cluster_loss * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)
        residual_l1 = torch.abs(pred_residual - target_residual).sum(dim=-1)
        residual_loss = (residual_l1 * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)
        stop_loss_raw = self.bce(pred_stop, target_stop)
        stop_loss = (stop_loss_raw * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)
        total = self.cluster_weight * cluster_loss + self.residual_weight * residual_loss + self.stop_weight * stop_loss
        return total, cluster_loss.detach(), residual_loss.detach(), stop_loss.detach()


def build_sample_from_row(row, prefix_points):
    points = row['_poly_clean']
    if len(points) <= prefix_points:
        return None
    ts = int(row['TIMESTAMP'])
    hour = (ts // 3600) % 24
    weekday = (ts // 86400) % 7
    return Sample(points=points, hour=hour, weekday=weekday, prefix_points=prefix_points)


def prepare_samples(df, prefix_points=20, max_samples=5000, max_total_points=96, min_future_points=3, seed=42):
    samples = []
    for _, row in df.iterrows():
        sample = build_sample_from_row(row, prefix_points=prefix_points)
        if sample is None:
            continue
        if len(sample.points) > max_total_points:
            sample = Sample(points=sample.points[:max_total_points], hour=sample.hour, weekday=sample.weekday, prefix_points=sample.prefix_points)
        if len(sample.points) - prefix_points < min_future_points:
            continue
        samples.append(sample)
    rng = np.random.default_rng(seed)
    if len(samples) > max_samples:
        idx = rng.choice(len(samples), size=max_samples, replace=False)
        samples = [samples[i] for i in idx]
    return samples


def build_aux_features(step_idx, total_points, hour, weekday, residual_x, residual_y):
    hour_angle = 2.0 * math.pi * hour / 24.0
    weekday_angle = 2.0 * math.pi * weekday / 7.0
    progress = step_idx / max(total_points - 1, 1)
    return [residual_x, residual_y, math.sin(hour_angle), math.cos(hour_angle), math.sin(weekday_angle), math.cos(weekday_angle), progress]


def sample_to_training_arrays(sample, tokenizer):
    points = sample.points
    point_info = [tokenizer.encode_point(lon, lat) for lon, lat in points]
    input_len = len(points) - 1
    cluster_ids = []
    aux_features = []
    target_cluster = []
    target_residual = []
    target_stop = []
    loss_mask = []
    for i in range(input_len):
        curr_cid, curr_rx, curr_ry = point_info[i]
        next_cid, next_rx, next_ry = point_info[i + 1]
        cluster_ids.append(curr_cid)
        aux_features.append(build_aux_features(i, len(points), sample.hour, sample.weekday, curr_rx, curr_ry))
        target_cluster.append(next_cid)
        target_residual.append([next_rx, next_ry])
        target_stop.append(1.0 if i == input_len - 1 else 0.0)
        loss_mask.append(1.0 if i >= sample.prefix_points - 1 else 0.0)
    return {
        'cluster_ids': np.asarray(cluster_ids, dtype=np.int64),
        'aux_features': np.asarray(aux_features, dtype=np.float32),
        'target_cluster': np.asarray(target_cluster, dtype=np.int64),
        'target_residual': np.asarray(target_residual, dtype=np.float32),
        'target_stop': np.asarray(target_stop, dtype=np.float32),
        'loss_mask': np.asarray(loss_mask, dtype=np.float32),
        'meta': {
            'observed_duration': (sample.prefix_points - 1) * 15,
            'true_duration': (len(points) - 1) * 15,
            'true_remaining_steps': len(points) - sample.prefix_points,
            'prefix_points': sample.prefix_points,
            'hour': sample.hour,
            'weekday': sample.weekday,
            'true_points': points,
        },
    }


def build_sequences(samples, tokenizer):
    return [sample_to_training_arrays(sample, tokenizer) for sample in samples]


def collate_batch(batch):
    max_len = max(len(item['cluster_ids']) for item in batch)
    aux_dim = batch[0]['aux_features'].shape[1]
    bs = len(batch)
    cluster_ids = np.zeros((bs, max_len), dtype=np.int64)
    aux_features = np.zeros((bs, max_len, aux_dim), dtype=np.float32)
    target_cluster = np.zeros((bs, max_len), dtype=np.int64)
    target_residual = np.zeros((bs, max_len, 2), dtype=np.float32)
    target_stop = np.zeros((bs, max_len), dtype=np.float32)
    loss_mask = np.zeros((bs, max_len), dtype=np.float32)
    padding_mask = np.ones((bs, max_len), dtype=bool)
    metas = []
    for i, item in enumerate(batch):
        length = len(item['cluster_ids'])
        cluster_ids[i, :length] = item['cluster_ids']
        aux_features[i, :length] = item['aux_features']
        target_cluster[i, :length] = item['target_cluster']
        target_residual[i, :length] = item['target_residual']
        target_stop[i, :length] = item['target_stop']
        loss_mask[i, :length] = item['loss_mask']
        padding_mask[i, :length] = False
        metas.append(item['meta'])
    return {
        'cluster_ids': torch.tensor(cluster_ids, dtype=torch.long),
        'aux_features': torch.tensor(aux_features, dtype=torch.float32),
        'target_cluster': torch.tensor(target_cluster, dtype=torch.long),
        'target_residual': torch.tensor(target_residual, dtype=torch.float32),
        'target_stop': torch.tensor(target_stop, dtype=torch.float32),
        'loss_mask': torch.tensor(loss_mask, dtype=torch.float32),
        'padding_mask': torch.tensor(padding_mask, dtype=torch.bool),
        'metas': metas,
    }


def evaluate_loader(model, loader, loss_computer, device):
    model.eval()
    total_loss = total_cluster = total_residual = total_stop = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            cluster_ids = batch['cluster_ids'].to(device)
            aux_features = batch['aux_features'].to(device)
            target_cluster = batch['target_cluster'].to(device)
            target_residual = batch['target_residual'].to(device)
            target_stop = batch['target_stop'].to(device)
            loss_mask = batch['loss_mask'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            pred_cluster, pred_residual, pred_stop = model(cluster_ids, aux_features, padding_mask=padding_mask)
            loss, cluster_loss, residual_loss, stop_loss = loss_computer(pred_cluster, pred_residual, pred_stop, target_cluster, target_residual, target_stop, loss_mask)
            total_loss += float(loss.item())
            total_cluster += float(cluster_loss.item())
            total_residual += float(residual_loss.item())
            total_stop += float(stop_loss.item())
            n_batches += 1
    denom = max(n_batches, 1)
    return {'loss': total_loss / denom, 'cluster_loss': total_cluster / denom, 'residual_loss': total_residual / denom, 'stop_loss': total_stop / denom}


def train_model(train_ds, val_ds, aux_dim, max_len, vocab_size, args):
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    model = ClusterGenerativeTransformer(vocab_size=vocab_size, aux_dim=aux_dim, d_model=args.d_model, nhead=args.nhead, num_layers=args.layers, dropout=args.dropout, max_len=max_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_computer = LossComputer(cluster_weight=args.cluster_weight, residual_weight=args.residual_weight, stop_weight=args.stop_weight, stop_pos_weight=args.stop_pos_weight)
    if device.type != 'cpu':
        loss_computer.bce = loss_computer.bce.to(device)
    best_val = float('inf')
    best_state = None
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_cluster = total_residual = total_stop = 0.0
        n_batches = 0
        for batch in train_loader:
            cluster_ids = batch['cluster_ids'].to(device)
            aux_features = batch['aux_features'].to(device)
            target_cluster = batch['target_cluster'].to(device)
            target_residual = batch['target_residual'].to(device)
            target_stop = batch['target_stop'].to(device)
            loss_mask = batch['loss_mask'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            optimizer.zero_grad()
            pred_cluster, pred_residual, pred_stop = model(cluster_ids, aux_features, padding_mask=padding_mask)
            loss, cluster_loss, residual_loss, stop_loss = loss_computer(pred_cluster, pred_residual, pred_stop, target_cluster, target_residual, target_stop, loss_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            total_cluster += float(cluster_loss.item())
            total_residual += float(residual_loss.item())
            total_stop += float(stop_loss.item())
            n_batches += 1
        train_stats = {'loss': total_loss / max(n_batches, 1), 'cluster_loss': total_cluster / max(n_batches, 1), 'residual_loss': total_residual / max(n_batches, 1), 'stop_loss': total_stop / max(n_batches, 1)}
        val_stats = evaluate_loader(model, val_loader, loss_computer, device)
        print(f"Epoch {epoch:02d} | train loss {train_stats['loss']:.4f} (cluster {train_stats['cluster_loss']:.4f}, residual {train_stats['residual_loss']:.4f}, stop {train_stats['stop_loss']:.4f}) | val loss {val_stats['loss']:.4f} (cluster {val_stats['cluster_loss']:.4f}, residual {val_stats['residual_loss']:.4f}, stop {val_stats['stop_loss']:.4f})")
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


def build_prefix_tokens(points, hour, weekday, tokenizer, aux_standardizer):
    cluster_ids = []
    aux = []
    total_points = len(points)
    for i, (lon, lat) in enumerate(points):
        cid, rx, ry = tokenizer.encode_point(lon, lat)
        cluster_ids.append(cid)
        aux.append(build_aux_features(i, total_points, hour, weekday, rx, ry))
    aux = aux_standardizer.transform(np.asarray(aux, dtype=np.float32))
    return np.asarray(cluster_ids, dtype=np.int64), aux


def sample_next_cluster(logits, temperature):
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    probs = F.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def rollout_one(model, sample_meta, tokenizer, aux_standardizer, residual_standardizer, device, stop_threshold=0.5, max_gen_steps=80, temperature=0.0):
    model.eval()
    points = [list(pt) for pt in sample_meta['true_points'][:sample_meta['prefix_points']]]
    hour = sample_meta['hour']
    weekday = sample_meta['weekday']
    generated_steps = 0
    stop_probs = []
    stopped = False
    for _ in range(max_gen_steps):
        cluster_ids_np, aux_np = build_prefix_tokens(points, hour, weekday, tokenizer, aux_standardizer)
        cluster_ids = torch.tensor(cluster_ids_np[None, :], dtype=torch.long, device=device)
        aux_features = torch.tensor(aux_np[None, :, :], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred_cluster, pred_residual, pred_stop = model(cluster_ids, aux_features, padding_mask=None)
            cluster_logits = pred_cluster[0, -1]
            next_cluster = sample_next_cluster(cluster_logits, temperature)
            pred_residual_std = pred_residual[0, -1].detach().cpu().numpy()[None, :]
            next_residual = residual_standardizer.inverse_transform(pred_residual_std)[0]
            stop_logit = pred_stop[0, -1].item()
        if temperature > 0:
            stop_prob = torch.sigmoid(torch.tensor(stop_logit / max(temperature, 1e-6))).item()
            stop_decision = np.random.rand() < stop_prob
        else:
            stop_prob = torch.sigmoid(torch.tensor(stop_logit)).item()
            stop_decision = stop_prob >= stop_threshold
        next_lon, next_lat = tokenizer.decode_point(next_cluster, float(next_residual[0]), float(next_residual[1]))
        points.append([next_lon, next_lat])
        generated_steps += 1
        stop_probs.append(stop_prob)
        if stop_decision:
            stopped = True
            break
    pred_duration = sample_meta['observed_duration'] + generated_steps * 15
    return {'pred_duration': pred_duration, 'generated_steps': generated_steps, 'stopped': stopped, 'stop_probs': stop_probs}


def run_rollout_eval(model, test_sequences, tokenizer, aux_standardizer, residual_standardizer, device, args, temperature=0.0, num_rollouts=1, label='greedy'):
    y_true, y_pred, true_remaining, pred_remaining, stop_biases = [], [], [], [], []
    stop_triggered = 0.0
    for item in test_sequences:
        rollouts = [rollout_one(model, item['meta'], tokenizer, aux_standardizer, residual_standardizer, device, stop_threshold=args.stop_threshold, max_gen_steps=args.max_gen_steps, temperature=temperature) for _ in range(num_rollouts)]
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
    return {'label': label, 'temperature': temperature, 'num_rollouts': num_rollouts, 'duration_mae': duration_stats['MAE'], 'duration_rmse': duration_stats['RMSE'], 'duration_mape': duration_stats['MAPE'], 'avg_generated_steps': float(np.mean(pred_remaining)), 'avg_true_remaining_steps': float(np.mean(true_remaining)), 'remaining_steps_mae': float(remaining_mae), 'remaining_steps_rmse': float(remaining_rmse), 'mean_stop_step_bias': float(np.mean(stop_biases)), 'mean_abs_stop_step_bias': float(np.mean(np.abs(stop_biases))), 'stop_trigger_rate': float(stop_triggered / max(len(test_sequences), 1)), 'n_test': len(test_sequences)}


def write_summary(args, clean_stats, counts, tokenizer, metrics_list, output_path):
    lines = ['# Taxi Generative Cluster Duration Prototype', '', '## Setup', f"- data: `{args.data}`", f"- raw rows loaded: {clean_stats.get('raw_rows', 'n/a')}", f"- cleaned rows kept: {clean_stats.get('after_clean', 'n/a')}", f"- sampled usable trips: {counts['all']}", f"- split: train={counts['train']}, val={counts['val']}, test={counts['test']}", f"- prefix_points: {args.prefix_points}", f"- max_total_points: {args.max_total_points}", f"- max_gen_steps: {args.max_gen_steps}", f"- tokenizer: MiniBatchKMeans with K={tokenizer.n_clusters}", '', '## Design', '- cluster token over learned spatial centers from train split only', '- per-step heads: next_cluster_id softmax + residual dx/dy regression + EOS logit', '- rollout duration = observed prefix duration + generated_steps × 15s', '- stochastic decoding applies temperature to cluster softmax and stop logit; residuals stay deterministic', '', '## Evaluation']
    for metrics in metrics_list:
        lines.extend([f"### {metrics['label']}", f"- temperature: {metrics['temperature']}", f"- num_rollouts: {metrics['num_rollouts']}", f"- duration MAE: {metrics['duration_mae']:.2f} s", f"- duration RMSE: {metrics['duration_rmse']:.2f} s", f"- duration MAPE: {metrics['duration_mape']:.2f}%", f"- avg generated remaining steps: {metrics['avg_generated_steps']:.2f}", f"- avg true remaining steps: {metrics['avg_true_remaining_steps']:.2f}", f"- remaining steps MAE: {metrics['remaining_steps_mae']:.2f}", f"- remaining steps RMSE: {metrics['remaining_steps_rmse']:.2f}", f"- mean stop-step bias: {metrics['mean_stop_step_bias']:.2f}", f"- mean abs stop-step bias: {metrics['mean_abs_stop_step_bias']:.2f}", f"- stop trigger rate: {metrics['stop_trigger_rate']:.3f}", ''])
    lines.extend(['## Command', '```bash', args.run_command, '```'])
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='kaggle_dl/train.csv')
    parser.add_argument('--n-sample', type=int, default=7000)
    parser.add_argument('--max-samples', type=int, default=5000)
    parser.add_argument('--prefix-points', type=int, default=20)
    parser.add_argument('--max-total-points', type=int, default=96)
    parser.add_argument('--min-future-points', type=int, default=3)
    parser.add_argument('--test-size', type=float, default=0.15)
    parser.add_argument('--val-size', type=float, default=0.15)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=96)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--cluster-weight', type=float, default=1.0)
    parser.add_argument('--residual-weight', type=float, default=2.0)
    parser.add_argument('--stop-weight', type=float, default=1.0)
    parser.add_argument('--stop-pos-weight', type=float, default=4.0)
    parser.add_argument('--stop-threshold', type=float, default=0.5)
    parser.add_argument('--max-gen-steps', type=int, default=80)
    parser.add_argument('--n-clusters', type=int, default=256)
    parser.add_argument('--stochastic-temperature', type=float, default=2.0)
    parser.add_argument('--stochastic-rollouts', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    args.run_command = '/home/test/anaconda3/bin/python taxi_generative_cluster_duration.py ' + ' '.join(sys.argv[1:])
    set_seed(args.seed)
    data_path = args.data
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), args.data)
    print('=' * 60)
    print('Taxi generative cluster duration prototype')
    print('=' * 60)
    print(f'Loading real data from: {data_path}')
    df, clean_stats = load_real_data(data_path, n_sample=args.n_sample)
    print('Building trajectory samples...')
    samples = prepare_samples(df, prefix_points=args.prefix_points, max_samples=args.max_samples, max_total_points=args.max_total_points, min_future_points=args.min_future_points, seed=args.seed)
    if len(samples) < 300:
        raise RuntimeError(f'Usable samples too few: {len(samples)}')
    train_samples, test_samples = train_test_split(samples, test_size=args.test_size, random_state=args.seed)
    train_samples, val_samples = train_test_split(train_samples, test_size=args.val_size, random_state=args.seed)
    print('Fitting cluster tokenizer on train split points only...')
    tokenizer = ClusterTokenizer(n_clusters=args.n_clusters, random_state=args.seed).fit(train_samples)
    train_sequences = build_sequences(train_samples, tokenizer)
    val_sequences = build_sequences(val_samples, tokenizer)
    test_sequences = build_sequences(test_samples, tokenizer)
    aux_standardizer = Standardizer().fit([item['aux_features'] for item in train_sequences])
    residual_standardizer = Standardizer().fit([item['target_residual'] for item in train_sequences])
    for seq_list in [train_sequences, val_sequences, test_sequences]:
        for item in seq_list:
            item['aux_features'] = aux_standardizer.transform(item['aux_features'])
            item['target_residual'] = residual_standardizer.transform(item['target_residual'])
    aux_dim = train_sequences[0]['aux_features'].shape[1]
    max_len = max(len(item['cluster_ids']) for item in train_sequences + val_sequences + test_sequences)
    counts = {'all': len(samples), 'train': len(train_sequences), 'val': len(val_sequences), 'test': len(test_sequences)}
    print(f"Usable samples: total={counts['all']}, train={counts['train']}, val={counts['val']}, test={counts['test']}, aux_dim={aux_dim}, max_len={max_len}, clusters={args.n_clusters}")
    train_ds = TrajectoryDataset(train_sequences)
    val_ds = TrajectoryDataset(val_sequences)
    model, device = train_model(train_ds, val_ds, aux_dim, max_len, args.n_clusters, args)
    print('Running autoregressive rollout on test set...')
    metrics_list = []
    greedy_metrics = run_rollout_eval(model, test_sequences, tokenizer, aux_standardizer, residual_standardizer, device, args, temperature=0.0, num_rollouts=1, label='greedy_t0')
    metrics_list.append(greedy_metrics)
    if args.stochastic_temperature > 0 and args.stochastic_rollouts > 1:
        stochastic_metrics = run_rollout_eval(model, test_sequences, tokenizer, aux_standardizer, residual_standardizer, device, args, temperature=args.stochastic_temperature, num_rollouts=args.stochastic_rollouts, label=f'stochastic_t{args.stochastic_temperature}_mean{args.stochastic_rollouts}')
        metrics_list.append(stochastic_metrics)
    print('\nRollout metrics')
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
    summary_path = os.path.join(OUTPUT_DIR, 'taxi_generative_cluster_summary.md')
    write_summary(args, clean_stats, counts, tokenizer, metrics_list, summary_path)
    print(f'Summary written to: {summary_path}')


if __name__ == '__main__':
    main()
