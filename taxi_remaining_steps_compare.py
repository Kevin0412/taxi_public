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
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from taxi_duration_prediction import (
    load_real_data,
    evaluate_predictions,
    OUTPUT_DIR,
    PORTO_BBOX,
    haversine,
)


GRID_LAT_DEG = 0.0022
GRID_LON_DEG = 0.0028


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


class ContinuousDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class DiscreteAuxDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class ContinuousEncoderRegressor(nn.Module):
    def __init__(self, token_dim, d_model=96, nhead=4, num_layers=2, dropout=0.1, max_len=32):
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
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, tokens, padding_mask=None):
        seq_len = tokens.size(1)
        pos = self.pos_embed[:, :seq_len, :]
        x = self.token_proj(tokens) + pos
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = self.norm(x)
        if padding_mask is None:
            pooled = x.mean(dim=1)
        else:
            keep = (~padding_mask).float().unsqueeze(-1)
            pooled = (x * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)
        return self.head(pooled).squeeze(-1)


class DiscreteAuxEncoderRegressor(nn.Module):
    def __init__(self, vocab_size, aux_dim, d_model=96, nhead=4, num_layers=2, dropout=0.1, max_len=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
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
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, token_ids, aux_features, padding_mask=None):
        seq_len = token_ids.size(1)
        pos = self.pos_embed[:, :seq_len, :]
        x = self.embed(token_ids) + self.aux_proj(aux_features) + pos
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = self.norm(x)
        if padding_mask is None:
            pooled = x.mean(dim=1)
        else:
            keep = (~padding_mask).float().unsqueeze(-1)
            pooled = (x * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)
        return self.head(pooled).squeeze(-1)


class GridTokenizer:
    def __init__(self, cell_lat=GRID_LAT_DEG, cell_lon=GRID_LON_DEG):
        self.lat_min = PORTO_BBOX['lat_min']
        self.lat_max = PORTO_BBOX['lat_max']
        self.lon_min = PORTO_BBOX['lon_min']
        self.lon_max = PORTO_BBOX['lon_max']
        self.cell_lat = cell_lat
        self.cell_lon = cell_lon
        self.n_lat = int(math.ceil((self.lat_max - self.lat_min) / self.cell_lat))
        self.n_lon = int(math.ceil((self.lon_max - self.lon_min) / self.cell_lon))
        self.vocab_size = self.n_lat * self.n_lon

    def encode_point(self, lon, lat):
        lat_pos = (lat - self.lat_min) / self.cell_lat
        lon_pos = (lon - self.lon_min) / self.cell_lon
        lat_idx = int(np.clip(np.floor(lat_pos), 0, self.n_lat - 1))
        lon_idx = int(np.clip(np.floor(lon_pos), 0, self.n_lon - 1))
        offset_y = float(np.clip(lat_pos - lat_idx, 0.0, 0.999999))
        offset_x = float(np.clip(lon_pos - lon_idx, 0.0, 0.999999))
        gid = lat_idx * self.n_lon + lon_idx
        return gid, offset_x, offset_y


class ClusterTokenizer:
    def __init__(self, n_clusters=256, batch_size=4096, random_state=42, max_fit_points=100000):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.random_state = random_state
        self.max_fit_points = max_fit_points
        self.kmeans = None
        self.centers = None

    def fit(self, samples):
        pts = []
        for sample in samples:
            pts.extend(sample.points[:sample.prefix_points])
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


FOURIER_SCALES = [0.01, 0.02]


def build_fourier_features(value):
    feats = []
    for scale in FOURIER_SCALES:
        angle = 2.0 * math.pi * value / scale
        feats.extend([math.sin(angle), math.cos(angle)])
    return feats


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
            sample = Sample(
                points=sample.points[:max_total_points],
                hour=sample.hour,
                weekday=sample.weekday,
                prefix_points=sample.prefix_points,
            )
        if len(sample.points) - prefix_points < min_future_points:
            continue
        samples.append(sample)
    rng = np.random.default_rng(seed)
    if len(samples) > max_samples:
        idx = rng.choice(len(samples), size=max_samples, replace=False)
        samples = [samples[i] for i in idx]
    return samples


def hour_weekday_features(hour, weekday):
    hour_angle = 2.0 * math.pi * hour / 24.0
    weekday_angle = 2.0 * math.pi * weekday / 7.0
    return [math.sin(hour_angle), math.cos(hour_angle), math.sin(weekday_angle), math.cos(weekday_angle)]


def continuous_point_features(points, idx, hour, weekday, prefix_points):
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
    progress = idx / max(prefix_points - 1, 1)
    feats = [
        rel_dx,
        rel_dy,
        delta_x,
        delta_y,
        seg_dist / 100.0,
        speed / 15.0,
        progress,
    ]
    feats.extend(hour_weekday_features(hour, weekday))
    feats.extend(build_fourier_features(rel_dx))
    feats.extend(build_fourier_features(rel_dy))
    return feats


def build_continuous_items(samples):
    items = []
    for sample in samples:
        prefix = sample.points[:sample.prefix_points]
        tokens = np.asarray(
            [continuous_point_features(prefix, i, sample.hour, sample.weekday, sample.prefix_points) for i in range(sample.prefix_points)],
            dtype=np.float32,
        )
        remaining_steps = float(len(sample.points) - sample.prefix_points)
        observed_duration = float((sample.prefix_points - 1) * 15)
        true_duration = float((len(sample.points) - 1) * 15)
        items.append({
            'tokens': tokens,
            'target': remaining_steps,
            'meta': {
                'observed_duration': observed_duration,
                'true_duration': true_duration,
                'true_remaining_steps': remaining_steps,
            },
        })
    return items


def build_discrete_aux(sample, tokenizer):
    token_ids = []
    aux = []
    prefix = sample.points[:sample.prefix_points]
    time_feats = hour_weekday_features(sample.hour, sample.weekday)
    for i, (lon, lat) in enumerate(prefix):
        token_id, x1, x2 = tokenizer.encode_point(lon, lat)
        progress = i / max(sample.prefix_points - 1, 1)
        token_ids.append(token_id)
        aux.append([x1, x2, *time_feats, progress])
    remaining_steps = float(len(sample.points) - sample.prefix_points)
    observed_duration = float((sample.prefix_points - 1) * 15)
    true_duration = float((len(sample.points) - 1) * 15)
    return {
        'token_ids': np.asarray(token_ids, dtype=np.int64),
        'aux_features': np.asarray(aux, dtype=np.float32),
        'target': remaining_steps,
        'meta': {
            'observed_duration': observed_duration,
            'true_duration': true_duration,
            'true_remaining_steps': remaining_steps,
        },
    }


def build_grid_items(samples, tokenizer):
    return [build_discrete_aux(sample, tokenizer) for sample in samples]


def build_cluster_items(samples, tokenizer):
    return [build_discrete_aux(sample, tokenizer) for sample in samples]


def standardize_continuous(train_items, other_item_groups):
    scaler = Standardizer().fit([item['tokens'] for item in train_items])
    for group in [train_items] + list(other_item_groups):
        for item in group:
            item['tokens'] = scaler.transform(item['tokens'])
    return scaler


def standardize_aux(train_items, other_item_groups):
    scaler = Standardizer().fit([item['aux_features'] for item in train_items])
    for group in [train_items] + list(other_item_groups):
        for item in group:
            item['aux_features'] = scaler.transform(item['aux_features'])
    return scaler


def collate_continuous(batch):
    max_len = max(item['tokens'].shape[0] for item in batch)
    token_dim = batch[0]['tokens'].shape[1]
    bs = len(batch)
    tokens = np.zeros((bs, max_len, token_dim), dtype=np.float32)
    padding_mask = np.ones((bs, max_len), dtype=bool)
    targets = np.zeros(bs, dtype=np.float32)
    metas = []
    for i, item in enumerate(batch):
        length = item['tokens'].shape[0]
        tokens[i, :length] = item['tokens']
        padding_mask[i, :length] = False
        targets[i] = item['target']
        metas.append(item['meta'])
    return {
        'tokens': torch.tensor(tokens, dtype=torch.float32),
        'padding_mask': torch.tensor(padding_mask, dtype=torch.bool),
        'targets': torch.tensor(targets, dtype=torch.float32),
        'metas': metas,
    }


def collate_discrete_aux(batch):
    max_len = max(item['token_ids'].shape[0] for item in batch)
    aux_dim = batch[0]['aux_features'].shape[1]
    bs = len(batch)
    token_ids = np.zeros((bs, max_len), dtype=np.int64)
    aux_features = np.zeros((bs, max_len, aux_dim), dtype=np.float32)
    padding_mask = np.ones((bs, max_len), dtype=bool)
    targets = np.zeros(bs, dtype=np.float32)
    metas = []
    for i, item in enumerate(batch):
        length = item['token_ids'].shape[0]
        token_ids[i, :length] = item['token_ids']
        aux_features[i, :length] = item['aux_features']
        padding_mask[i, :length] = False
        targets[i] = item['target']
        metas.append(item['meta'])
    return {
        'token_ids': torch.tensor(token_ids, dtype=torch.long),
        'aux_features': torch.tensor(aux_features, dtype=torch.float32),
        'padding_mask': torch.tensor(padding_mask, dtype=torch.bool),
        'targets': torch.tensor(targets, dtype=torch.float32),
        'metas': metas,
    }


def run_epoch_continuous(model, loader, optimizer, loss_fn, device, train=True):
    model.train(train)
    total_loss = 0.0
    preds = []
    targets = []
    for batch in loader:
        tokens = batch['tokens'].to(device)
        padding_mask = batch['padding_mask'].to(device)
        y = batch['targets'].to(device)
        if train:
            optimizer.zero_grad()
        pred = model(tokens, padding_mask=padding_mask)
        loss = loss_fn(pred, y)
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += float(loss.item()) * y.size(0)
        preds.append(pred.detach().cpu().numpy())
        targets.append(y.detach().cpu().numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets)
    return {
        'loss': total_loss / max(len(loader.dataset), 1),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': mean_squared_error(y_true, y_pred) ** 0.5,
    }


def run_epoch_discrete_aux(model, loader, optimizer, loss_fn, device, train=True):
    model.train(train)
    total_loss = 0.0
    preds = []
    targets = []
    for batch in loader:
        token_ids = batch['token_ids'].to(device)
        aux_features = batch['aux_features'].to(device)
        padding_mask = batch['padding_mask'].to(device)
        y = batch['targets'].to(device)
        if train:
            optimizer.zero_grad()
        pred = model(token_ids, aux_features, padding_mask=padding_mask)
        loss = loss_fn(pred, y)
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += float(loss.item()) * y.size(0)
        preds.append(pred.detach().cpu().numpy())
        targets.append(y.detach().cpu().numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets)
    return {
        'loss': total_loss / max(len(loader.dataset), 1),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': mean_squared_error(y_true, y_pred) ** 0.5,
    }


def train_representation(name, train_ds, val_ds, test_ds, model, collate_fn, args, discrete=False):
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(beta=args.huber_beta)
    run_epoch = run_epoch_discrete_aux if discrete else run_epoch_continuous

    best_state = None
    best_val = float('inf')
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, optimizer, loss_fn, device, train=True)
        with torch.no_grad():
            val_stats = run_epoch(model, val_loader, optimizer, loss_fn, device, train=False)
        print(
            f"[{name}] epoch {epoch:02d} | train loss {train_stats['loss']:.4f} mae {train_stats['mae']:.3f} rmse {train_stats['rmse']:.3f} | "
            f"val loss {val_stats['loss']:.4f} mae {val_stats['mae']:.3f} rmse {val_stats['rmse']:.3f}"
        )
        if val_stats['loss'] < best_val - 1e-4:
            best_val = val_stats['loss']
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[{name}] early stopping at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = evaluate_test(model, test_loader, device, discrete=discrete)
    return metrics


def evaluate_test(model, loader, device, discrete=False):
    model.eval()
    remaining_true = []
    remaining_pred = []
    duration_true = []
    duration_pred = []
    with torch.no_grad():
        for batch in loader:
            y = batch['targets'].numpy()
            metas = batch['metas']
            if discrete:
                pred = model(
                    batch['token_ids'].to(device),
                    batch['aux_features'].to(device),
                    padding_mask=batch['padding_mask'].to(device),
                )
            else:
                pred = model(
                    batch['tokens'].to(device),
                    padding_mask=batch['padding_mask'].to(device),
                )
            pred = pred.detach().cpu().numpy()
            pred = np.clip(pred, 0.0, None)
            remaining_true.extend(y.tolist())
            remaining_pred.extend(pred.tolist())
            for yy, pp, meta in zip(y.tolist(), pred.tolist(), metas):
                duration_true.append(meta['true_duration'])
                duration_pred.append(meta['observed_duration'] + pp * 15.0)

    remaining_true = np.asarray(remaining_true)
    remaining_pred = np.asarray(remaining_pred)
    duration_true = np.asarray(duration_true)
    duration_pred = np.asarray(duration_pred)
    duration_stats = evaluate_predictions(duration_true, duration_pred)
    return {
        'remaining_steps_mae': float(mean_absolute_error(remaining_true, remaining_pred)),
        'remaining_steps_rmse': float(mean_squared_error(remaining_true, remaining_pred) ** 0.5),
        'duration_mae': float(duration_stats['MAE']),
        'duration_rmse': float(duration_stats['RMSE']),
        'duration_mape': float(duration_stats['MAPE']),
        'avg_true_remaining_steps': float(np.mean(remaining_true)),
        'avg_pred_remaining_steps': float(np.mean(remaining_pred)),
        'n_test': int(len(remaining_true)),
    }


def write_summary(args, clean_stats, counts, results, output_path):
    best_remaining = min(results, key=lambda x: x['metrics']['remaining_steps_mae'])
    best_duration = min(results, key=lambda x: x['metrics']['duration_mae'])
    lines = [
        '# Taxi Remaining Steps Compare',
        '',
        '## Setup',
        f"- data: `{args.data}`",
        f"- raw rows loaded: {clean_stats.get('raw_rows', 'n/a')}",
        f"- cleaned rows kept: {clean_stats.get('after_clean', 'n/a')}",
        f"- sampled usable trips: {counts['all']}",
        f"- split: train={counts['train']}, val={counts['val']}, test={counts['test']}",
        f"- prefix_points: {args.prefix_points}",
        f"- n_sample: {args.n_sample}",
        f"- max_samples: {args.max_samples}",
        f"- model: encoder-only Transformer direct regression to remaining steps",
        f"- duration conversion: observed_prefix_duration + predicted_remaining_steps × 15s",
        '',
        '## Representations',
        '- A: continuous point tokens',
        '- B: fixed-grid token + offset',
        '- C: cluster token + residual (fit on train split only)',
        '',
        '## Results',
        '| Representation | Remaining MAE | Remaining RMSE | Total Duration MAE (s) | Total Duration RMSE (s) | Total Duration MAPE | Avg Pred Remaining | Avg True Remaining |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for result in results:
        m = result['metrics']
        lines.append(
            f"| {result['name']} | {m['remaining_steps_mae']:.3f} | {m['remaining_steps_rmse']:.3f} | {m['duration_mae']:.2f} | {m['duration_rmse']:.2f} | {m['duration_mape']:.2f}% | {m['avg_pred_remaining_steps']:.3f} | {m['avg_true_remaining_steps']:.3f} |"
        )
    lines.extend([
        '',
        '## Notes',
        f"- best remaining-steps MAE: {best_remaining['name']} ({best_remaining['metrics']['remaining_steps_mae']:.3f})",
        f"- best total-duration MAE: {best_duration['name']} ({best_duration['metrics']['duration_mae']:.2f} s)",
        '',
        '## Command',
        '```bash',
        args.run_command,
        '```',
    ])
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='kaggle_dl/train.csv')
    parser.add_argument('--n-sample', type=int, default=8000)
    parser.add_argument('--max-samples', type=int, default=5000)
    parser.add_argument('--prefix-points', type=int, default=20)
    parser.add_argument('--max-total-points', type=int, default=96)
    parser.add_argument('--min-future-points', type=int, default=3)
    parser.add_argument('--test-size', type=float, default=0.15)
    parser.add_argument('--val-size', type=float, default=0.15)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--d-model', type=int, default=96)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--huber-beta', type=float, default=2.0)
    parser.add_argument('--grid-lat-deg', type=float, default=GRID_LAT_DEG)
    parser.add_argument('--grid-lon-deg', type=float, default=GRID_LON_DEG)
    parser.add_argument('--n-clusters', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    args.run_command = '/home/test/anaconda3/bin/python taxi_remaining_steps_compare.py ' + ' '.join(sys.argv[1:])
    set_seed(args.seed)

    data_path = args.data
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), args.data)

    print('=' * 60)
    print('Taxi remaining-steps direct prediction compare')
    print('=' * 60)
    print(f'Loading real data from: {data_path}')
    df, clean_stats = load_real_data(data_path, n_sample=args.n_sample)

    print('Building samples...')
    samples = prepare_samples(
        df,
        prefix_points=args.prefix_points,
        max_samples=args.max_samples,
        max_total_points=args.max_total_points,
        min_future_points=args.min_future_points,
        seed=args.seed,
    )
    if len(samples) < 300:
        raise RuntimeError(f'Usable samples too few: {len(samples)}')

    train_samples, test_samples = train_test_split(samples, test_size=args.test_size, random_state=args.seed)
    train_samples, val_samples = train_test_split(train_samples, test_size=args.val_size, random_state=args.seed)
    counts = {
        'all': len(samples),
        'train': len(train_samples),
        'val': len(val_samples),
        'test': len(test_samples),
    }
    print(
        f"Usable samples: total={counts['all']}, train={counts['train']}, val={counts['val']}, test={counts['test']}, prefix_points={args.prefix_points}"
    )

    results = []

    print('\nPreparing continuous representation...')
    cont_train = build_continuous_items(train_samples)
    cont_val = build_continuous_items(val_samples)
    cont_test = build_continuous_items(test_samples)
    standardize_continuous(cont_train, [cont_val, cont_test])
    cont_model = ContinuousEncoderRegressor(
        token_dim=cont_train[0]['tokens'].shape[1],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dropout=args.dropout,
        max_len=args.prefix_points,
    )
    cont_metrics = train_representation(
        'continuous',
        ContinuousDataset(cont_train),
        ContinuousDataset(cont_val),
        ContinuousDataset(cont_test),
        cont_model,
        collate_continuous,
        args,
        discrete=False,
    )
    results.append({'name': 'A. continuous point tokens', 'metrics': cont_metrics})

    print('\nPreparing fixed-grid representation...')
    grid_tokenizer = GridTokenizer(cell_lat=args.grid_lat_deg, cell_lon=args.grid_lon_deg)
    grid_train = build_grid_items(train_samples, grid_tokenizer)
    grid_val = build_grid_items(val_samples, grid_tokenizer)
    grid_test = build_grid_items(test_samples, grid_tokenizer)
    standardize_aux(grid_train, [grid_val, grid_test])
    grid_model = DiscreteAuxEncoderRegressor(
        vocab_size=grid_tokenizer.vocab_size,
        aux_dim=grid_train[0]['aux_features'].shape[1],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dropout=args.dropout,
        max_len=args.prefix_points,
    )
    grid_metrics = train_representation(
        'grid',
        DiscreteAuxDataset(grid_train),
        DiscreteAuxDataset(grid_val),
        DiscreteAuxDataset(grid_test),
        grid_model,
        collate_discrete_aux,
        args,
        discrete=True,
    )
    results.append({'name': 'B. fixed-grid token + offset', 'metrics': grid_metrics})

    print('\nPreparing cluster representation...')
    cluster_tokenizer = ClusterTokenizer(n_clusters=args.n_clusters, random_state=args.seed).fit(train_samples)
    cluster_train = build_cluster_items(train_samples, cluster_tokenizer)
    cluster_val = build_cluster_items(val_samples, cluster_tokenizer)
    cluster_test = build_cluster_items(test_samples, cluster_tokenizer)
    standardize_aux(cluster_train, [cluster_val, cluster_test])
    cluster_model = DiscreteAuxEncoderRegressor(
        vocab_size=args.n_clusters,
        aux_dim=cluster_train[0]['aux_features'].shape[1],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dropout=args.dropout,
        max_len=args.prefix_points,
    )
    cluster_metrics = train_representation(
        'cluster',
        DiscreteAuxDataset(cluster_train),
        DiscreteAuxDataset(cluster_val),
        DiscreteAuxDataset(cluster_test),
        cluster_model,
        collate_discrete_aux,
        args,
        discrete=True,
    )
    results.append({'name': 'C. cluster token + residual', 'metrics': cluster_metrics})

    print('\nFinal comparison')
    for result in results:
        m = result['metrics']
        print(
            f"{result['name']}: remaining MAE={m['remaining_steps_mae']:.3f}, remaining RMSE={m['remaining_steps_rmse']:.3f}, "
            f"duration MAE={m['duration_mae']:.2f}s, duration RMSE={m['duration_rmse']:.2f}s, duration MAPE={m['duration_mape']:.2f}%"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUTPUT_DIR, 'taxi_remaining_steps_compare.md')
    write_summary(args, clean_stats, counts, results, summary_path)
    print(f'Summary written to: {summary_path}')


if __name__ == '__main__':
    main()
