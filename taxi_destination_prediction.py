#!/usr/bin/env python3
"""Porto taxi destination prediction.

Task 3: predict trip destination from a trajectory prefix.

This script:
- loads the first N rows from Kaggle Porto Taxi Trajectory data
- parses and cleans POLYLINE trajectories
- builds prefix features for 20/40/60% of each trajectory
- compares two models:
  A. Ridge regression on destination coordinates
  B. RandomForest classification on a 0.002° destination grid
- evaluates endpoint Haversine error
- writes plots and a markdown summary to the output directory
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False


PORTO_BBOX = {
    "lat_min": 41.0,
    "lat_max": 41.3,
    "lon_min": -8.7,
    "lon_max": -8.5,
}
GRID_SIZE = 0.002  # degrees
TIME_STEP_SECONDS = 15.0
SEQ_SAMPLE_LIMIT = 3500
SEQ_EPOCHS = 30
SEQ_PATIENCE = 5
SEQ_BATCH_SIZE = 128
SEQ_MAX_LEN = 30
SEQ_D_MODEL = 96
DEFAULT_INPUT = os.path.expanduser("~/taxi/kaggle_dl/train.csv")
DEFAULT_OUTPUT_DIR = os.path.expanduser("~/taxi/output/task3")


@dataclass
class TrajectorySample:
    points: List[Tuple[float, float]]  # [(lon, lat), ...]
    timestamp: int
    prefix_ratio: float
    prefix_points: int
    true_end_lon: float
    true_end_lat: float


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return Haversine distance in kilometers."""
    r = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def lonlat_to_local_xy(lon, lat, ref_lat=41.157):
    """Convert lon,lat to local xy in meters."""
    R = 6371000.0
    x = (lon - (-8.616)) * math.cos(math.radians(ref_lat)) * (math.pi / 180.0) * R
    y = (lat - ref_lat) * (math.pi / 180.0) * R
    return x, y


class OSMFeatureExtractor:
    ROAD_FLAGS = {
        'major': 1,
        'secondary': 2,
        'tertiary': 4,
        'local': 8,
        'nonmotorized': 16,
    }
    DOMINANT_ORDER = ['major', 'secondary', 'tertiary', 'local', 'nonmotorized']
    DOMINANT_ENC = {name: i + 1 for i, name in enumerate(DOMINANT_ORDER)}
    CELL_SIZE_M = 120.0

    def __init__(self, osm_json_path):
        self.osm_json_path = osm_json_path
        self.query_cache = {}
        self._load()

    @classmethod
    def map_highway_class(cls, highway):
        if not highway:
            return None
        highway = str(highway)
        if highway in {'motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary', 'primary_link'}:
            return 'major'
        if highway in {'secondary', 'secondary_link'}:
            return 'secondary'
        if highway in {'tertiary', 'tertiary_link'}:
            return 'tertiary'
        if highway in {'residential', 'service', 'living_street', 'unclassified'}:
            return 'local'
        if highway in {'pedestrian', 'footway', 'cycleway', 'path', 'track', 'steps'}:
            return 'nonmotorized'
        return None

    def _cell_key(self, x, y):
        return int(x // self.CELL_SIZE_M), int(y // self.CELL_SIZE_M)

    def _neighbor_candidate_indices(self, x, y, radius_m):
        hop = max(1, int(math.ceil(radius_m / self.CELL_SIZE_M)))
        cx, cy = self._cell_key(x, y)
        out = []
        for dx in range(-hop, hop + 1):
            for dy in range(-hop, hop + 1):
                out.extend(self.grid.get((cx + dx, cy + dy), []))
        if not out:
            return np.empty(0, dtype=np.int32)
        return np.asarray(out, dtype=np.int32)

    def _nearest_distance_for_flag(self, x, y, flag_name, search_radius_m=1500.0):
        idx = self._neighbor_candidate_indices(x, y, search_radius_m)
        if len(idx) == 0:
            return search_radius_m
        mask = (self.node_masks[idx] & self.ROAD_FLAGS[flag_name]) != 0
        if not np.any(mask):
            return search_radius_m
        dx = self.node_x[idx[mask]] - x
        dy = self.node_y[idx[mask]] - y
        return float(np.sqrt(np.min(dx * dx + dy * dy)))

    def point_features(self, lon, lat, prefix):
        cache_key = (prefix, round(lon, 5), round(lat, 5))
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]

        x, y = lonlat_to_local_xy(lon, lat)
        idx_200 = self._neighbor_candidate_indices(x, y, 200.0)
        idx_300 = self._neighbor_candidate_indices(x, y, 300.0)

        def count_in_radius(indices, radius_m, flag_name=None):
            if len(indices) == 0:
                return 0
            dx = self.node_x[indices] - x
            dy = self.node_y[indices] - y
            inside = (dx * dx + dy * dy) <= radius_m * radius_m
            if flag_name is not None:
                inside &= (self.node_masks[indices] & self.ROAD_FLAGS[flag_name]) != 0
            return int(np.sum(inside))

        dom_counts = {}
        for name in self.DOMINANT_ORDER:
            dom_counts[name] = count_in_radius(idx_200, 200.0, name)
        dominant_type = max(self.DOMINANT_ORDER, key=lambda name: (dom_counts[name], -self.DOMINANT_ENC[name]))
        dominant_enc = self.DOMINANT_ENC[dominant_type] if dom_counts[dominant_type] > 0 else 0

        features = {
            f'{prefix}_nearest_major_m': self._nearest_distance_for_flag(x, y, 'major'),
            f'{prefix}_nearest_secondary_m': self._nearest_distance_for_flag(x, y, 'secondary'),
            f'{prefix}_local_road_count_r200': count_in_radius(idx_200, 200.0, 'local'),
            f'{prefix}_major_road_count_r300': count_in_radius(idx_300, 300.0, 'major'),
            f'{prefix}_node_density_r200': count_in_radius(idx_200, 200.0),
            f'{prefix}_dominant_road_type_enc': dominant_enc,
        }
        self.query_cache[cache_key] = features
        return features

    def _load(self):
        print(f"加载 OSM 路网特征：{self.osm_json_path}")
        with open(self.osm_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        node_coords = {}
        node_masks = defaultdict(int)
        way_count = 0

        for el in data.get('elements', []):
            if el.get('type') == 'node':
                lon = el.get('lon')
                lat = el.get('lat')
                if lon is None or lat is None:
                    continue
                if point_in_porto(lon, lat):
                    node_coords[el['id']] = (float(lon), float(lat))

        for el in data.get('elements', []):
            if el.get('type') != 'way':
                continue
            road_class = self.map_highway_class((el.get('tags') or {}).get('highway'))
            if road_class is None:
                continue
            flag = self.ROAD_FLAGS[road_class]
            valid_nodes = 0
            for node_id in el.get('nodes', []):
                if node_id in node_coords:
                    node_masks[node_id] |= flag
                    valid_nodes += 1
            if valid_nodes >= 2:
                way_count += 1

        kept_ids = [node_id for node_id, mask in node_masks.items() if mask > 0 and node_id in node_coords]
        xs, ys, masks = [], [], []
        grid = defaultdict(list)
        for idx, node_id in enumerate(kept_ids):
            lon, lat = node_coords[node_id]
            x, y = lonlat_to_local_xy(lon, lat)
            xs.append(x)
            ys.append(y)
            mask = node_masks[node_id]
            masks.append(mask)
            grid[self._cell_key(x, y)].append(idx)

        self.node_x = np.asarray(xs, dtype=np.float32)
        self.node_y = np.asarray(ys, dtype=np.float32)
        self.node_masks = np.asarray(masks, dtype=np.int32)
        self.grid = grid
        self.node_count = len(kept_ids)
        self.way_count = way_count
        print(f"OSM 特征索引完成：{self.node_count} 个道路节点，{self.way_count} 条道路 way")


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    y = math.sin(math.radians(lon2 - lon1)) * math.cos(math.radians(lat2))
    x = (
        math.cos(math.radians(lat1)) * math.sin(math.radians(lat2))
        - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(lon2 - lon1))
    )
    deg = math.degrees(math.atan2(y, x))
    return (deg + 360.0) % 360.0


def angle_diff_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def point_in_porto(lon: float, lat: float) -> bool:
    return (
        PORTO_BBOX["lon_min"] <= lon <= PORTO_BBOX["lon_max"]
        and PORTO_BBOX["lat_min"] <= lat <= PORTO_BBOX["lat_max"]
    )


def parse_polyline(polyline_value) -> Optional[List[Tuple[float, float]]]:
    """Parse POLYLINE and keep only Porto-valid coordinates."""
    if isinstance(polyline_value, list):
        raw = polyline_value
    else:
        try:
            raw = json.loads(polyline_value)
        except Exception:
            try:
                raw = ast.literal_eval(polyline_value)
            except Exception:
                return None

    if not isinstance(raw, list):
        return None

    points: List[Tuple[float, float]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            return None
        try:
            lon = float(item[0])
            lat = float(item[1])
        except Exception:
            return None
        if not (np.isfinite(lon) and np.isfinite(lat)):
            return None
        if not point_in_porto(lon, lat):
            return None
        points.append((lon, lat))

    if len(points) < 10:
        return None
    return points


# -----------------------------------------------------------------------------
# Grid helpers
# -----------------------------------------------------------------------------

def build_grid_cfg() -> Dict[str, float]:
    lat_bins = int(math.ceil((PORTO_BBOX["lat_max"] - PORTO_BBOX["lat_min"]) / GRID_SIZE))
    lon_bins = int(math.ceil((PORTO_BBOX["lon_max"] - PORTO_BBOX["lon_min"]) / GRID_SIZE))
    return {
        "lat_min": PORTO_BBOX["lat_min"],
        "lon_min": PORTO_BBOX["lon_min"],
        "lat_bins": lat_bins,
        "lon_bins": lon_bins,
        "grid_size": GRID_SIZE,
    }


def destination_to_grid_id(lat: float, lon: float, cfg: Dict[str, float]) -> int:
    lat_idx = int(math.floor((lat - cfg["lat_min"]) / cfg["grid_size"]))
    lon_idx = int(math.floor((lon - cfg["lon_min"]) / cfg["grid_size"]))
    lat_idx = max(0, min(cfg["lat_bins"] - 1, lat_idx))
    lon_idx = max(0, min(cfg["lon_bins"] - 1, lon_idx))
    return lat_idx * cfg["lon_bins"] + lon_idx


def grid_id_to_center(grid_id: int, cfg: Dict[str, float]) -> Tuple[float, float]:
    lat_idx = int(grid_id // cfg["lon_bins"])
    lon_idx = int(grid_id % cfg["lon_bins"])
    lat = cfg["lat_min"] + (lat_idx + 0.5) * cfg["grid_size"]
    lon = cfg["lon_min"] + (lon_idx + 0.5) * cfg["grid_size"]
    lat = max(PORTO_BBOX["lat_min"], min(PORTO_BBOX["lat_max"], lat))
    lon = max(PORTO_BBOX["lon_min"], min(PORTO_BBOX["lon_max"], lon))
    return lat, lon


# -----------------------------------------------------------------------------
# Data loading and cleaning
# -----------------------------------------------------------------------------

def load_and_clean(csv_path: str, n_sample: int, max_clean_samples: int = 6000) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = pd.read_csv(csv_path, nrows=n_sample)
    stats = {
        "raw_rows": int(len(df)),
        "missing_false_rows": 0,
        "parsed_valid_rows": 0,
        "kept_rows": 0,
        "dropped_bad_polyline": 0,
        "dropped_too_short": 0,
        "dropped_invalid_bounds": 0,
    }

    if "MISSING_DATA" in df.columns:
        df = df[df["MISSING_DATA"] == False].copy()  # noqa: E712
    stats["missing_false_rows"] = int(len(df))

    records = []
    for _, row in df.iterrows():
        pts = parse_polyline(row["POLYLINE"])
        if pts is None:
            stats["dropped_bad_polyline"] += 1
            continue

        if len(pts) < 10:
            stats["dropped_too_short"] += 1
            continue

        # Require the cleaned trajectory to stay inside the Porto bbox.
        if any(not point_in_porto(lon, lat) for lon, lat in pts):
            stats["dropped_invalid_bounds"] += 1
            continue

        rec = row.to_dict()
        rec["_points"] = pts
        rec["_end_lon"] = pts[-1][0]
        rec["_end_lat"] = pts[-1][1]
        rec["_start_lon"] = pts[0][0]
        rec["_start_lat"] = pts[0][1]
        records.append(rec)
        stats["parsed_valid_rows"] += 1

    clean_df = pd.DataFrame(records)
    if len(clean_df) > max_clean_samples:
        clean_df = clean_df.sample(n=max_clean_samples, random_state=42).reset_index(drop=True)
    else:
        clean_df = clean_df.reset_index(drop=True)
    stats["kept_rows"] = int(len(clean_df))
    return clean_df, stats


# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------

def trajectory_prefix(points: Sequence[Tuple[float, float]], ratio: float) -> Tuple[List[Tuple[float, float]], int]:
    prefix_points = max(2, int(len(points) * ratio))
    prefix_points = min(prefix_points, len(points) - 1)
    return list(points[:prefix_points]), prefix_points


def prefix_features(prefix: Sequence[Tuple[float, float]], row: pd.Series, osm_extractor=None) -> Dict[str, float]:
    """Build the required feature set for a single prefix."""
    start_lon, start_lat = prefix[0]
    end_lon, end_lat = prefix[-1]
    dists_km = [haversine_km(prefix[i - 1][1], prefix[i - 1][0], prefix[i][1], prefix[i][0]) for i in range(1, len(prefix))]
    prefix_dist_km = float(np.sum(dists_km)) if dists_km else 0.0
    prefix_duration_s = float((len(prefix) - 1) * TIME_STEP_SECONDS)
    prefix_avg_speed = prefix_dist_km / (prefix_duration_s / 3600.0) if prefix_duration_s > 0 else 0.0
    br = bearing_deg(start_lat, start_lon, end_lat, end_lon)
    segment_bearings = [
        bearing_deg(prefix[i - 1][1], prefix[i - 1][0], prefix[i][1], prefix[i][0]) for i in range(1, len(prefix))
    ]
    bearing_std = float(np.std(segment_bearings)) if len(segment_bearings) > 1 else 0.0
    straight_line_dist = haversine_km(start_lat, start_lon, end_lat, end_lon)
    detour_ratio = prefix_dist_km / straight_line_dist if straight_line_dist > 1e-9 else 1.0
    hour = int((int(row["TIMESTAMP"]) // 3600) % 24)
    weekday = int((int(row["TIMESTAMP"]) // 86400) % 7)
    displacement_x = end_lon - start_lon
    displacement_y = end_lat - start_lat

    feat = {
        "start_lat": start_lat,
        "start_lon": start_lon,
        "end_lat": end_lat,
        "end_lon": end_lon,
        "prefix_dist": prefix_dist_km,
        "prefix_duration": prefix_duration_s,
        "prefix_avg_speed": prefix_avg_speed,
        "bearing": br,
        "bearing_std": bearing_std,
        "straight_line_dist": straight_line_dist,
        "detour_ratio": detour_ratio,
        "hour_of_day": hour,
        "weekday": weekday,
        "displacement_x": displacement_x,
        "displacement_y": displacement_y,
    }
    if osm_extractor is not None:
        feat.update(osm_extractor.point_features(start_lon, start_lat, 'start'))
        feat.update(osm_extractor.point_features(end_lon, end_lat, 'end'))
    return feat


def build_seq_tokens(points: Sequence[Tuple[float, float]], row: pd.Series, seq_max_len: int = SEQ_MAX_LEN, osm_extractor=None) -> Tuple[np.ndarray, np.ndarray]:
    """Build sequence tokens and meta features for SeqTransformer."""
    if len(points) < 2:
        raise ValueError("trajectory prefix needs at least 2 points")

    start_lon, start_lat = points[0]
    total_points = len(points)
    total_dist_km = 0.0
    total_turn = 0.0
    token_rows: List[List[float]] = []

    prev_bearing = None
    cumulative_km = 0.0
    for idx, (lon, lat) in enumerate(points):
        if idx == 0:
            step_km = 0.0
            step_bearing = 0.0
            turn_angle = 0.0
        else:
            plon, plat = points[idx - 1]
            step_km = haversine_km(plat, plon, lat, lon)
            cumulative_km += step_km
            step_bearing = bearing_deg(plat, plon, lat, lon)
            if prev_bearing is None:
                turn_angle = 0.0
            else:
                turn_angle = angle_diff_deg(prev_bearing, step_bearing)
            prev_bearing = step_bearing
            total_dist_km += step_km
            total_turn += turn_angle

        dx_start = lon - start_lon
        dy_start = lat - start_lat
        progress = idx / max(1, total_points - 1)
        straight_km = haversine_km(start_lat, start_lon, lat, lon)
        local_speed = step_km / (TIME_STEP_SECONDS / 3600.0) if idx > 0 else 0.0
        token_rows.append([
            dx_start,
            dy_start,
            step_km,
            math.cos(math.radians(step_bearing)) if idx > 0 else 1.0,
            math.sin(math.radians(step_bearing)) if idx > 0 else 0.0,
            local_speed,
            turn_angle,
            progress,
            cumulative_km,
            straight_km,
        ])

    token_arr = np.asarray(token_rows, dtype=np.float32)
    if len(token_arr) > seq_max_len:
        idx = np.linspace(0, len(token_arr) - 1, seq_max_len, dtype=int)
        token_arr = token_arr[idx]
    elif len(token_arr) < seq_max_len:
        pad = np.zeros((seq_max_len - len(token_arr), token_arr.shape[1]), dtype=np.float32)
        token_arr = np.concatenate([token_arr, pad], axis=0)

    prefix_duration_s = float((len(points) - 1) * TIME_STEP_SECONDS)
    prefix_avg_speed = total_dist_km / (prefix_duration_s / 3600.0) if prefix_duration_s > 0 else 0.0
    hour = int((int(row["TIMESTAMP"]) // 3600) % 24)
    weekday = int((int(row["TIMESTAMP"]) // 86400) % 7)
    meta_items = [
        hour / 23.0,
        weekday / 6.0,
        min(1.0, len(points) / float(seq_max_len)),
        prefix_avg_speed / 60.0,
    ]
    if osm_extractor is not None:
        feat = prefix_features(points, row, osm_extractor=osm_extractor)
        osm_keys = [
            'start_nearest_major_m',
            'start_nearest_secondary_m',
            'start_local_road_count_r200',
            'start_major_road_count_r300',
            'start_node_density_r200',
            'start_dominant_road_type_enc',
            'end_nearest_major_m',
            'end_nearest_secondary_m',
            'end_local_road_count_r200',
            'end_major_road_count_r300',
            'end_node_density_r200',
            'end_dominant_road_type_enc',
        ]
        meta_items.extend([float(feat[k]) for k in osm_keys])
    meta = np.asarray(meta_items, dtype=np.float32)
    return token_arr, meta


def build_feature_frame(df: pd.DataFrame, prefix_ratio: float, grid_cfg: Dict[str, float], osm_extractor=None) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        points = row["_points"]
        prefix, prefix_points = trajectory_prefix(points, prefix_ratio)
        feat = prefix_features(prefix, row, osm_extractor=osm_extractor)
        feat["prefix_ratio"] = prefix_ratio
        feat["prefix_points"] = prefix_points
        feat["true_end_lat"] = row["_end_lat"]
        feat["true_end_lon"] = row["_end_lon"]
        feat["grid_id"] = destination_to_grid_id(row["_end_lat"], row["_end_lon"], grid_cfg)
        rows.append(feat)
    return pd.DataFrame(rows)


def build_seq_dataset(df: pd.DataFrame, prefix_ratio: float, seq_max_len: int = SEQ_MAX_LEN, osm_extractor=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    token_rows = []
    meta_rows = []
    y_rows = []
    for _, row in df.iterrows():
        points = row["_points"]
        prefix, _ = trajectory_prefix(points, prefix_ratio)
        tok, meta = build_seq_tokens(prefix, row, seq_max_len=seq_max_len, osm_extractor=osm_extractor)
        token_rows.append(tok)
        meta_rows.append(meta)
        y_rows.append([row["_end_lat"], row["_end_lon"]])
    return np.asarray(token_rows, dtype=np.float32), np.asarray(meta_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.float32)


def collect_prefix_points(df: pd.DataFrame, prefix_ratio: float) -> np.ndarray:
    points = []
    for _, row in df.iterrows():
        prefix, _ = trajectory_prefix(row["_points"], prefix_ratio)
        points.extend(prefix)
    if not points:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


class GridTokenizer:
    def __init__(self, cell_lat: float = 0.0018, cell_lon: float = 0.0022):
        self.lat_min = PORTO_BBOX["lat_min"]
        self.lat_max = PORTO_BBOX["lat_max"]
        self.lon_min = PORTO_BBOX["lon_min"]
        self.lon_max = PORTO_BBOX["lon_max"]
        self.cell_lat = float(cell_lat)
        self.cell_lon = float(cell_lon)
        self.n_lat = int(math.ceil((self.lat_max - self.lat_min) / self.cell_lat))
        self.n_lon = int(math.ceil((self.lon_max - self.lon_min) / self.cell_lon))
        self.vocab_size = int(self.n_lat * self.n_lon)

    def encode_point(self, lon: float, lat: float) -> Tuple[int, float, float]:
        lat_pos = (lat - self.lat_min) / self.cell_lat
        lon_pos = (lon - self.lon_min) / self.cell_lon
        lat_idx = int(np.clip(np.floor(lat_pos), 0, self.n_lat - 1))
        lon_idx = int(np.clip(np.floor(lon_pos), 0, self.n_lon - 1))
        offset_y = float(np.clip(lat_pos - lat_idx, 0.0, 0.999999))
        offset_x = float(np.clip(lon_pos - lon_idx, 0.0, 0.999999))
        grid_id = lat_idx * self.n_lon + lon_idx
        return grid_id, offset_x, offset_y


class ClusterTokenizer:
    def __init__(self, n_clusters: int = 256, batch_size: int = 4096, random_state: int = 42, max_fit_points: int = 100000):
        self.n_clusters = int(n_clusters)
        self.batch_size = int(batch_size)
        self.random_state = int(random_state)
        self.max_fit_points = int(max_fit_points)
        self.kmeans = None
        self.centers = None

    def fit(self, prefix_points: np.ndarray):
        points = np.asarray(prefix_points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError('prefix_points must have shape (n, 2)')
        if len(points) == 0:
            raise ValueError('prefix_points is empty')
        if len(points) > self.max_fit_points:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(points), size=self.max_fit_points, replace=False)
            points = points[idx]
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=self.batch_size,
            random_state=self.random_state,
            n_init=5,
        )
        self.kmeans.fit(points)
        self.centers = self.kmeans.cluster_centers_.astype(np.float32)
        return self

    def encode_point(self, lon: float, lat: float) -> Tuple[int, float, float]:
        if self.centers is None:
            raise RuntimeError('ClusterTokenizer must be fit before encoding')
        point = np.asarray([lon, lat], dtype=np.float32)
        d2 = np.sum((self.centers - point[None, :]) ** 2, axis=1)
        cluster_id = int(np.argmin(d2))
        center = self.centers[cluster_id]
        residual = point - center
        return cluster_id, float(residual[0]), float(residual[1])


def _build_token_sequence(points: Sequence[Tuple[float, float]], row: pd.Series, tokenizer, seq_max_len: int = SEQ_MAX_LEN, osm_extractor=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(points) < 2:
        raise ValueError('trajectory prefix needs at least 2 points')

    prefix_feat = prefix_features(points, row, osm_extractor=osm_extractor)
    hour = float(prefix_feat['hour_of_day'])
    weekday = float(prefix_feat['weekday'])
    prefix_len = float(len(points))
    token_ids = []
    aux_rows = []

    for idx, (lon, lat) in enumerate(points):
        raw_id, off_x, off_y = tokenizer.encode_point(lon, lat)
        token_ids.append(int(raw_id) + 1)  # reserve 0 for padding
        progress = idx / max(1, len(points) - 1)
        aux_rows.append([off_x, off_y, hour, weekday, progress])

    token_arr = np.asarray(token_ids, dtype=np.int64)
    aux_arr = np.asarray(aux_rows, dtype=np.float32)
    if len(token_arr) > seq_max_len:
        idx = np.linspace(0, len(token_arr) - 1, seq_max_len, dtype=int)
        token_arr = token_arr[idx]
        aux_arr = aux_arr[idx]
    elif len(token_arr) < seq_max_len:
        pad = np.zeros((seq_max_len - len(token_arr), aux_arr.shape[1]), dtype=np.float32)
        token_arr = np.concatenate([token_arr, np.zeros(seq_max_len - len(token_arr), dtype=np.int64)], axis=0)
        aux_arr = np.concatenate([aux_arr, pad], axis=0)

    osm_keys = ['start_nearest_major_m', 'start_nearest_secondary_m', 'start_local_road_count_r200',
                'start_major_road_count_r300', 'start_node_density_r200', 'start_dominant_road_type_enc',
                'end_nearest_major_m', 'end_nearest_secondary_m', 'end_local_road_count_r200',
                'end_major_road_count_r300', 'end_node_density_r200', 'end_dominant_road_type_enc']
    meta_values = [
        prefix_feat['hour_of_day'],
        prefix_feat['weekday'],
        prefix_len,
        prefix_feat['prefix_avg_speed'],
        prefix_feat['straight_line_dist'],
        prefix_feat['detour_ratio'],
        prefix_feat['bearing'],
    ]
    for k in osm_keys:
        if k in prefix_feat:
            meta_values.append(prefix_feat[k])
    meta = np.asarray(meta_values, dtype=np.float32)
    y = np.asarray([row['_end_lat'], row['_end_lon']], dtype=np.float32)
    return token_arr, aux_arr, meta, y


def build_token_dataset(df: pd.DataFrame, prefix_ratio: float, tokenizer, seq_max_len: int = SEQ_MAX_LEN, osm_extractor=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    token_rows = []
    aux_rows = []
    meta_rows = []
    y_rows = []
    for _, row in df.iterrows():
        points = row['_points']
        prefix, _ = trajectory_prefix(points, prefix_ratio)
        token_arr, aux_arr, meta, y = _build_token_sequence(prefix, row, tokenizer, seq_max_len=seq_max_len, osm_extractor=osm_extractor)
        token_rows.append(token_arr)
        aux_rows.append(aux_arr)
        meta_rows.append(meta)
        y_rows.append(y)
    return (
        np.asarray(token_rows, dtype=np.int64),
        np.asarray(aux_rows, dtype=np.float32),
        np.asarray(meta_rows, dtype=np.float32),
        np.asarray(y_rows, dtype=np.float32),
    )


def build_grid_seq_dataset(df: pd.DataFrame, prefix_ratio: float, tokenizer: GridTokenizer, seq_max_len: int = SEQ_MAX_LEN, osm_extractor=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return build_token_dataset(df, prefix_ratio, tokenizer, seq_max_len=seq_max_len, osm_extractor=osm_extractor)


def build_cluster_seq_dataset(df: pd.DataFrame, prefix_ratio: float, tokenizer: ClusterTokenizer, seq_max_len: int = SEQ_MAX_LEN, osm_extractor=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return build_token_dataset(df, prefix_ratio, tokenizer, seq_max_len=seq_max_len, osm_extractor=osm_extractor)


# -----------------------------------------------------------------------------
# Modeling
# -----------------------------------------------------------------------------
# Modeling
# -----------------------------------------------------------------------------

def split_and_train_regression(X: pd.DataFrame, y: np.ndarray, seed: int = 42):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=2.0, random_state=seed)),
    ])
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return model, x_test, y_test, pred


def split_and_train_classification(X: pd.DataFrame, y: np.ndarray, seed: int = 42):
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=None
    )
    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=seed,
        class_weight="balanced_subsample",
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return model, x_test, y_test, pred


def haversine_errors_km(y_true_latlon: np.ndarray, y_pred_latlon: np.ndarray) -> np.ndarray:
    return np.array(
        [
            haversine_km(t[0], t[1], p[0], p[1])
            for t, p in zip(y_true_latlon, y_pred_latlon)
        ],
        dtype=float,
    )


def summarize_errors(errors_km: np.ndarray) -> Dict[str, float]:
    return {
        "mean_km": float(np.mean(errors_km)),
        "median_km": float(np.median(errors_km)),
        "max_km": float(np.max(errors_km)),
        "mae_km": float(np.mean(np.abs(errors_km))),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


if TORCH_AVAILABLE:

    class SeqTransformerDestinationRegressor(nn.Module):
        def __init__(self, token_dim: int, meta_dim: int, seq_len: int, d_model: int = SEQ_D_MODEL, nhead: int = 4, num_layers: int = 2, dropout: float = 0.12):
            super().__init__()
            self.token_proj = nn.Sequential(
                nn.Linear(token_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
            self.meta_proj = nn.Sequential(
                nn.Linear(meta_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))
            enc = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
            self.norm = nn.LayerNorm(d_model)
            self.pool_score = nn.Linear(d_model, 1)
            self.head = nn.Sequential(
                nn.Linear(d_model * 3, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 2),
            )

        def forward(self, tokens, meta):
            tok = self.token_proj(tokens)
            cls = self.cls_token.expand(tokens.size(0), -1, -1)
            x = torch.cat([cls, tok], dim=1)
            x = x + self.pos_embed[:, : x.size(1), :]
            x = self.encoder(x)
            cls_feat = self.norm(x[:, 0, :])
            tok_feat = self.norm(x[:, 1:, :])
            attn = torch.softmax(self.pool_score(tok_feat), dim=1)
            pooled = torch.sum(tok_feat * attn, dim=1)
            meta_feat = self.meta_proj(meta)
            out = self.head(torch.cat([cls_feat, pooled, meta_feat], dim=1))
            return out


    class DiscreteTokenDestinationRegressor(nn.Module):
        def __init__(self, vocab_size: int, aux_dim: int, meta_dim: int, seq_len: int, d_model: int = SEQ_D_MODEL, nhead: int = 4, num_layers: int = 2, dropout: float = 0.12):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
            self.aux_proj = nn.Sequential(
                nn.Linear(aux_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
            self.meta_proj = nn.Sequential(
                nn.Linear(meta_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))
            enc = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
            self.norm = nn.LayerNorm(d_model)
            self.pool_score = nn.Linear(d_model, 1)
            self.head = nn.Sequential(
                nn.Linear(d_model * 3, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 2),
            )

        def forward(self, token_ids, aux_features, meta):
            tok = self.embed(token_ids) + self.aux_proj(aux_features)
            cls = self.cls_token.expand(token_ids.size(0), -1, -1)
            x = torch.cat([cls, tok], dim=1)
            x = x + self.pos_embed[:, : x.size(1), :]
            padding_mask = torch.cat(
                [
                    torch.zeros((token_ids.size(0), 1), dtype=torch.bool, device=token_ids.device),
                    token_ids.eq(0),
                ],
                dim=1,
            )
            x = self.encoder(x, src_key_padding_mask=padding_mask)
            cls_feat = self.norm(x[:, 0, :])
            tok_feat = self.norm(x[:, 1:, :])
            tok_mask = token_ids.eq(0)
            attn_logits = self.pool_score(tok_feat).squeeze(-1)
            attn_logits = attn_logits.masked_fill(tok_mask, -1e9)
            attn = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
            pooled = torch.sum(tok_feat * attn, dim=1)
            meta_feat = self.meta_proj(meta)
            out = self.head(torch.cat([cls_feat, pooled, meta_feat], dim=1))
            return out


    def fit_seq_transformer(
        X_tokens: np.ndarray,
        X_meta: np.ndarray,
        y: np.ndarray,
        seed: int = 42,
        epochs: int = SEQ_EPOCHS,
        batch_size: int = SEQ_BATCH_SIZE,
        patience: int = SEQ_PATIENCE,
    ):
        if len(X_tokens) < 200:
            return None, None, None, None

        set_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"SeqTransformer device: {device}")

        if len(X_tokens) > SEQ_SAMPLE_LIMIT:
            idx = np.linspace(0, len(X_tokens) - 1, SEQ_SAMPLE_LIMIT, dtype=int)
            X_tokens = X_tokens[idx]
            X_meta = X_meta[idx]
            y = y[idx]
            print(f"SeqTransformer sample cap: {len(X_tokens)}")

        idx = np.arange(len(X_tokens))
        tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=seed)
        tr_idx, va_idx = train_test_split(tr_idx, test_size=0.1, random_state=seed)

        Xtr_tok, Xva_tok, Xte_tok = X_tokens[tr_idx], X_tokens[va_idx], X_tokens[te_idx]
        Xtr_meta, Xva_meta, Xte_meta = X_meta[tr_idx], X_meta[va_idx], X_meta[te_idx]
        ytr, yva, yte = y[tr_idx], y[va_idx], y[te_idx]

        if len(Xva_tok) < 32 or len(Xtr_tok) < 128:
            return None, None, None, None

        tok_mean = Xtr_tok.mean(axis=(0, 1), keepdims=True)
        tok_std = Xtr_tok.std(axis=(0, 1), keepdims=True) + 1e-6
        meta_mean = Xtr_meta.mean(axis=0, keepdims=True)
        meta_std = Xtr_meta.std(axis=0, keepdims=True) + 1e-6

        Xtr_tok = (Xtr_tok - tok_mean) / tok_std
        Xva_tok = (Xva_tok - tok_mean) / tok_std
        Xte_tok = (Xte_tok - tok_mean) / tok_std
        Xtr_meta = (Xtr_meta - meta_mean) / meta_std
        Xva_meta = (Xva_meta - meta_mean) / meta_std
        Xte_meta = (Xte_meta - meta_mean) / meta_std

        y_mean = ytr.mean(axis=0)
        y_std = ytr.std(axis=0)
        y_std = np.where(y_std < 1e-6, 1.0, y_std)
        ytr_norm = (ytr - y_mean) / y_std
        yva_norm = (yva - y_mean) / y_std
        yte_norm = (yte - y_mean) / y_std

        train_ds = TensorDataset(
            torch.tensor(Xtr_tok, dtype=torch.float32),
            torch.tensor(Xtr_meta, dtype=torch.float32),
            torch.tensor(ytr_norm, dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(Xva_tok, dtype=torch.float32),
            torch.tensor(Xva_meta, dtype=torch.float32),
            torch.tensor(yva_norm, dtype=torch.float32),
        )
        test_tok = torch.tensor(Xte_tok, dtype=torch.float32)
        test_meta = torch.tensor(Xte_meta, dtype=torch.float32)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

        model = SeqTransformerDestinationRegressor(
            token_dim=Xtr_tok.shape[2],
            meta_dim=Xtr_meta.shape[1],
            seq_len=Xtr_tok.shape[1],
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.6, patience=1)
        crit = nn.SmoothL1Loss(beta=0.20)

        best_state = None
        best_val = float("inf")
        wait = 0
        print(f"SeqTransformer train/val/test: {len(train_ds)}/{len(val_ds)}/{len(te_idx)}")
        for epoch in range(1, epochs + 1):
            model.train()
            tr_losses = []
            for batch_tok, batch_meta, batch_y in train_loader:
                batch_tok = batch_tok.to(device)
                batch_meta = batch_meta.to(device)
                batch_y = batch_y.to(device)
                opt.zero_grad(set_to_none=True)
                pred = model(batch_tok, batch_meta)
                loss = crit(pred, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tr_losses.append(float(loss.item()))

            model.eval()
            va_losses = []
            with torch.no_grad():
                for batch_tok, batch_meta, batch_y in val_loader:
                    batch_tok = batch_tok.to(device)
                    batch_meta = batch_meta.to(device)
                    batch_y = batch_y.to(device)
                    pred = model(batch_tok, batch_meta)
                    va_losses.append(float(crit(pred, batch_y).item()))
            tr_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")
            va_loss = float(np.mean(va_losses)) if va_losses else float("nan")
            print(f"  epoch {epoch:02d}: train={tr_loss:.4f} val={va_loss:.4f} lr={opt.param_groups[0]['lr']:.5f}")
            sched.step(va_loss)
            if va_loss < best_val - 1e-4:
                best_val = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print("  SeqTransformer early stopping.")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred_norm = model(test_tok.to(device), test_meta.to(device)).cpu().numpy()
        pred = pred_norm * y_std + y_mean
        pred = np.clip(pred, [PORTO_BBOX["lat_min"], PORTO_BBOX["lon_min"]], [PORTO_BBOX["lat_max"], PORTO_BBOX["lon_max"]])
        errors = haversine_errors_km(yte, pred)
        metrics = summarize_errors(errors)
        metrics.update({"n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(te_idx), "best_val_loss": float(best_val)})
        return model, pred, yte, metrics


    def fit_token_transformer(
        token_ids: np.ndarray,
        aux_features: np.ndarray,
        meta: np.ndarray,
        y: np.ndarray,
        seed: int = 42,
        epochs: int = SEQ_EPOCHS,
        batch_size: int = SEQ_BATCH_SIZE,
        patience: int = SEQ_PATIENCE,
    ):
        if len(token_ids) < 200:
            return None, None, None, None

        set_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"TokenTransformer device: {device}")

        if len(token_ids) > SEQ_SAMPLE_LIMIT:
            idx = np.linspace(0, len(token_ids) - 1, SEQ_SAMPLE_LIMIT, dtype=int)
            token_ids = token_ids[idx]
            aux_features = aux_features[idx]
            meta = meta[idx]
            y = y[idx]
            print(f"TokenTransformer sample cap: {len(token_ids)}")

        idx = np.arange(len(token_ids))
        tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=seed)
        tr_idx, va_idx = train_test_split(tr_idx, test_size=0.1, random_state=seed)

        Xtr_tok, Xva_tok, Xte_tok = token_ids[tr_idx], token_ids[va_idx], token_ids[te_idx]
        Xtr_aux, Xva_aux, Xte_aux = aux_features[tr_idx], aux_features[va_idx], aux_features[te_idx]
        Xtr_meta, Xva_meta, Xte_meta = meta[tr_idx], meta[va_idx], meta[te_idx]
        ytr, yva, yte = y[tr_idx], y[va_idx], y[te_idx]

        if len(Xva_tok) < 32 or len(Xtr_tok) < 128:
            return None, None, None, None

        aux_mean = Xtr_aux.mean(axis=(0, 1), keepdims=True)
        aux_std = Xtr_aux.std(axis=(0, 1), keepdims=True) + 1e-6
        meta_mean = Xtr_meta.mean(axis=0, keepdims=True)
        meta_std = Xtr_meta.std(axis=0, keepdims=True) + 1e-6

        Xtr_aux = (Xtr_aux - aux_mean) / aux_std
        Xva_aux = (Xva_aux - aux_mean) / aux_std
        Xte_aux = (Xte_aux - aux_mean) / aux_std
        Xtr_meta = (Xtr_meta - meta_mean) / meta_std
        Xva_meta = (Xva_meta - meta_mean) / meta_std
        Xte_meta = (Xte_meta - meta_mean) / meta_std

        y_mean = ytr.mean(axis=0)
        y_std = ytr.std(axis=0)
        y_std = np.where(y_std < 1e-6, 1.0, y_std)
        ytr_norm = (ytr - y_mean) / y_std
        yva_norm = (yva - y_mean) / y_std
        yte_norm = (yte - y_mean) / y_std

        train_ds = TensorDataset(
            torch.tensor(Xtr_tok, dtype=torch.long),
            torch.tensor(Xtr_aux, dtype=torch.float32),
            torch.tensor(Xtr_meta, dtype=torch.float32),
            torch.tensor(ytr_norm, dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(Xva_tok, dtype=torch.long),
            torch.tensor(Xva_aux, dtype=torch.float32),
            torch.tensor(Xva_meta, dtype=torch.float32),
            torch.tensor(yva_norm, dtype=torch.float32),
        )
        test_tok = torch.tensor(Xte_tok, dtype=torch.long)
        test_aux = torch.tensor(Xte_aux, dtype=torch.float32)
        test_meta = torch.tensor(Xte_meta, dtype=torch.float32)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

        model = DiscreteTokenDestinationRegressor(
            vocab_size=int(np.max(token_ids)) + 1,
            aux_dim=Xtr_aux.shape[2],
            meta_dim=Xtr_meta.shape[1],
            seq_len=Xtr_tok.shape[1],
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.6, patience=1)
        crit = nn.SmoothL1Loss(beta=0.20)

        best_state = None
        best_val = float("inf")
        wait = 0
        print(f"TokenTransformer train/val/test: {len(train_ds)}/{len(val_ds)}/{len(te_idx)}")
        for epoch in range(1, epochs + 1):
            model.train()
            tr_losses = []
            for batch_tok, batch_aux, batch_meta, batch_y in train_loader:
                batch_tok = batch_tok.to(device)
                batch_aux = batch_aux.to(device)
                batch_meta = batch_meta.to(device)
                batch_y = batch_y.to(device)
                opt.zero_grad(set_to_none=True)
                pred = model(batch_tok, batch_aux, batch_meta)
                loss = crit(pred, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tr_losses.append(float(loss.item()))

            model.eval()
            va_losses = []
            with torch.no_grad():
                for batch_tok, batch_aux, batch_meta, batch_y in val_loader:
                    batch_tok = batch_tok.to(device)
                    batch_aux = batch_aux.to(device)
                    batch_meta = batch_meta.to(device)
                    batch_y = batch_y.to(device)
                    pred = model(batch_tok, batch_aux, batch_meta)
                    va_losses.append(float(crit(pred, batch_y).item()))
            tr_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")
            va_loss = float(np.mean(va_losses)) if va_losses else float("nan")
            print(f"  epoch {epoch:02d}: train={tr_loss:.4f} val={va_loss:.4f} lr={opt.param_groups[0]['lr']:.5f}")
            sched.step(va_loss)
            if va_loss < best_val - 1e-4:
                best_val = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print("  TokenTransformer early stopping.")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred_norm = model(test_tok.to(device), test_aux.to(device), test_meta.to(device)).cpu().numpy()
        pred = pred_norm * y_std + y_mean
        pred = np.clip(pred, [PORTO_BBOX["lat_min"], PORTO_BBOX["lon_min"]], [PORTO_BBOX["lat_max"], PORTO_BBOX["lon_max"]])
        errors = haversine_errors_km(yte, pred)
        metrics = summarize_errors(errors)
        metrics.update({"n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(te_idx), "best_val_loss": float(best_val)})
        return model, pred, yte, metrics


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def save_line_plot(results: Dict[float, Dict[str, Dict[str, float]]], output_path: str) -> None:
    ratios = sorted(results.keys())
    reg_means = [results[r]["regression"]["mean_km"] for r in ratios]
    cls_means = [results[r]["classification"]["mean_km"] for r in ratios]
    seq_means = [results[r].get("seqtransformer", {}).get("mean_km", np.nan) for r in ratios]

    plt.figure(figsize=(8.5, 5.2))
    plt.plot([int(r * 100) for r in ratios], reg_means, marker="o", linewidth=2, label="Ridge regression")
    plt.plot([int(r * 100) for r in ratios], cls_means, marker="o", linewidth=2, label="Grid classification")
    if not np.all(np.isnan(seq_means)):
        plt.plot([int(r * 100) for r in ratios], seq_means, marker="o", linewidth=2, label="SeqTransformer")

    r60 = results.get(0.6, {})
    if "grid_token" in r60:
        plt.scatter([60], [r60["grid_token"]["mean_km"]], s=100, marker="^", label="Grid Token Transformer")
    if "cluster_token" in r60:
        plt.scatter([60], [r60["cluster_token"]["mean_km"]], s=100, marker="s", label="Cluster Token Transformer")

    plt.xlabel("Prefix length (% of trajectory)")
    plt.ylabel("Mean Haversine error (km)")
    plt.title("Prefix length vs destination error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_all_methods_comparison_plot(results: Dict[float, Dict[str, Dict[str, float]]], output_path: str) -> None:
    source = results.get(0.6, {})
    method_order = [
        ("regression", "Ridge regression", "#4c78a8"),
        ("classification", "Grid classification", "#f58518"),
        ("seqtransformer", "SeqTransformer", "#54a24b"),
        ("grid_token", "Grid Token Transformer", "#b279a2"),
        ("cluster_token", "Cluster Token Transformer", "#e45756"),
    ]
    labels = []
    values = []
    colors = []
    for key, label, color in method_order:
        if key in source:
            labels.append(label)
            values.append(source[key]["mean_km"])
            colors.append(color)

    if not values:
        return

    plt.figure(figsize=(9, 5.5))
    plt.bar(labels, values, color=colors)
    plt.ylabel("Mean Haversine error (km)")
    plt.title("All methods comparison at 60% prefix")
    plt.xticks(rotation=18, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_scatter_plot(
    true_latlon: np.ndarray,
    reg_pred: np.ndarray,
    cls_pred: np.ndarray,
    output_path: str,
    title_suffix: str = "",
    sample_n: int = 50,
):
    n = min(sample_n, len(true_latlon))
    idx = np.linspace(0, len(true_latlon) - 1, n, dtype=int)
    true_latlon = true_latlon[idx]
    reg_pred = reg_pred[idx]
    cls_pred = cls_pred[idx]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax, pred, title in [
        (axes[0], reg_pred, "Ridge regression"),
        (axes[1], cls_pred, "Grid classification"),
    ]:
        ax.scatter(true_latlon[:, 1], true_latlon[:, 0], s=35, c="black", label="True endpoint", alpha=0.8)
        ax.scatter(pred[:, 1], pred[:, 0], s=28, c="#d95f02", label="Predicted endpoint", alpha=0.75)
        for i in range(len(true_latlon)):
            ax.plot([true_latlon[i, 1], pred[i, 1]], [true_latlon[i, 0], pred[i, 0]], color="gray", alpha=0.25, linewidth=0.8)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)
        ax.set_xlim(PORTO_BBOX["lon_min"], PORTO_BBOX["lon_max"])
        ax.set_ylim(PORTO_BBOX["lat_min"], PORTO_BBOX["lat_max"])
    fig.suptitle(f"Predicted vs true destination points {title_suffix}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_histogram(errors_reg: np.ndarray, errors_cls: np.ndarray, errors_seq: Optional[np.ndarray], output_path: str) -> None:
    plt.figure(figsize=(8, 5))
    max_error = max(float(errors_reg.max()), float(errors_cls.max()))
    if errors_seq is not None and len(errors_seq) > 0:
        max_error = max(max_error, float(errors_seq.max()))
    bins = np.linspace(0, max_error, 32)
    plt.hist(errors_reg, bins=bins, alpha=0.55, label="Ridge regression")
    plt.hist(errors_cls, bins=bins, alpha=0.55, label="Grid classification")
    if errors_seq is not None and len(errors_seq) > 0:
        plt.hist(errors_seq, bins=bins, alpha=0.55, label="SeqTransformer")
    plt.xlabel("Haversine error (km)")
    plt.ylabel("Count")
    plt.title("Prediction error distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_feature_importance(model: RandomForestClassifier, feature_names: Sequence[str], output_path: str) -> None:
    importances = np.asarray(model.feature_importances_)
    order = np.argsort(importances)[::-1]
    top = order[: min(15, len(order))]

    plt.figure(figsize=(9, 6))
    plt.barh(np.array(feature_names)[top][::-1], importances[top][::-1], color="#4c78a8")
    plt.xlabel("Importance")
    plt.title("RandomForest feature importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


# -----------------------------------------------------------------------------
# Summary writing
# -----------------------------------------------------------------------------

def format_result_table(results: Dict[float, Dict[str, Dict[str, float]]]) -> str:
    lines = []
    lines.append("| Prefix | Method | Mean (km) | Median (km) | Max (km) |")
    lines.append("|---|---|---:|---:|---:|")
    for ratio in sorted(results.keys()):
        pct = int(ratio * 100)
        for method in ["regression", "classification", "seqtransformer"]:
            if method not in results[ratio]:
                continue
            s = results[ratio][method]
            label = {
                "regression": "Ridge regression",
                "classification": "Grid classification",
                "seqtransformer": "SeqTransformer",
            }[method]
            lines.append(
                f"| {pct}% | {label} | {s['mean_km']:.4f} | {s['median_km']:.4f} | {s['max_km']:.4f} |"
            )
    return "\n".join(lines)


def trend_analysis(results: Dict[float, Dict[str, Dict[str, float]]]) -> str:
    ratios = sorted(results.keys())
    reg = [results[r]["regression"]["mean_km"] for r in ratios]
    cls = [results[r]["classification"]["mean_km"] for r in ratios]
    seq = [results[r].get("seqtransformer", {}).get("mean_km", np.inf) for r in ratios]
    best_reg = ratios[int(np.argmin(reg))]
    best_cls = ratios[int(np.argmin(cls))]
    lines = [
        f"- Regression error is best at {int(best_reg * 100)}% prefix.",
        f"- Classification error is best at {int(best_cls * 100)}% prefix.",
    ]
    if np.isfinite(np.min(seq)):
        best_seq = ratios[int(np.argmin(seq))]
        lines.append(f"- SeqTransformer error is best at {int(best_seq * 100)}% prefix.")
    mean_candidates = {
        "Ridge regression": min(reg),
        "Grid classification": min(cls),
    }
    if np.isfinite(np.min(seq)):
        mean_candidates["SeqTransformer"] = min(seq)
    better = min(mean_candidates, key=mean_candidates.get)
    lines.append(f"- Overall best mean error comes from {better}.")
    lines.append("- Longer prefixes usually reduce error because the observed path contains more direction and displacement signal.")
    return "\n".join(lines)


def write_summary(
    output_dir: str,
    stats: Dict[str, int],
    results: Dict[float, Dict[str, Dict[str, float]]],
    class_counts: Dict[float, int],
    best_ratio: float,
    best_method: str,
    seq_note: str,
) -> str:
    summary_path = os.path.join(output_dir, "task3_destination_summary.md")
    table_md = format_result_table(results)
    analysis = trend_analysis(results)

    lines = []
    lines.append("# Task 3: Destination Prediction Summary")
    lines.append("")
    lines.append("## Data")
    lines.append(f"- raw rows loaded: {stats['raw_rows']}")
    lines.append(f"- rows after MISSING_DATA filter: {stats['missing_false_rows']}")
    lines.append(f"- valid cleaned trajectories: {stats['kept_rows']}")
    lines.append(f"- dropped invalid POLYLINE rows: {stats['dropped_bad_polyline']}")
    lines.append(f"- dropped too-short rows: {stats['dropped_too_short']}")
    lines.append(f"- dropped out-of-bounds rows: {stats['dropped_invalid_bounds']}")
    lines.append("")
    lines.append("## Feature set")
    lines.append("- start_lat, start_lon")
    lines.append("- end_lat, end_lon")
    lines.append("- prefix_dist")
    lines.append("- prefix_duration")
    lines.append("- prefix_avg_speed")
    lines.append("- bearing")
    lines.append("- bearing_std")
    lines.append("- straight_line_dist")
    lines.append("- detour_ratio")
    lines.append("- hour_of_day")
    lines.append("- weekday")
    lines.append("- displacement_x, displacement_y")
    lines.append("- SeqTransformer token features: dx/dy, step vector, local speed, turn angle, progress, cumulative distance")
    lines.append("- SeqTransformer meta features: hour, weekday, prefix length, average speed")
    lines.append("- Grid Token Transformer token features: grid_id, offset_x, offset_y, hour, weekday, progress")
    lines.append("- Cluster Token Transformer token features: cluster_id, residual_x, residual_y, hour, weekday, progress")
    lines.append("- Token transformer meta features: hour, weekday, prefix_len, avg_speed, straight_dist, detour_ratio, bearing")
    lines.append("")
    lines.append("## Grid classification")
    lines.append(f"- grid size: {GRID_SIZE:.3f}° × {GRID_SIZE:.3f}°")
    lines.append(f"- observed grid classes at 60% prefix set: {class_counts.get(0.6, 0)}")
    lines.append("")
    lines.append("## Token transformers")
    lines.append("- Grid tokenizer: cell_lat=0.0018, cell_lon=0.0022")
    lines.append("- Cluster tokenizer: n_clusters=256; fit only on training prefix points")
    lines.append("")
    lines.append("## Results table")
    lines.append(table_md)
    lines.append("")
    lines.append("## Best setting")
    lines.append(f"- best prefix ratio: {int(best_ratio * 100)}%")
    lines.append(f"- best method: {best_method}")
    lines.append("")
    lines.append("## Trend analysis")
    lines.append(analysis)
    lines.append("")
    lines.append("## Notes")
    lines.append("- Errors are computed with Haversine distance in kilometers.")
    lines.append("- The classification prediction is mapped to the center of the predicted destination grid cell.")
    if seq_note:
        lines.append(seq_note)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return summary_path


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def run(n_sample: int, output_dir: str, seed: int = 42, osm_json: Optional[str] = None) -> None:
    os.makedirs(output_dir, exist_ok=True)
    grid_cfg = build_grid_cfg()
    set_seed(seed)

    osm_extractor = None
    if osm_json is not None:
        osm_path = os.path.expanduser(osm_json)
        if not os.path.isabs(osm_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            candidate = os.path.join(script_dir, osm_path)
            if os.path.exists(candidate):
                osm_path = candidate
        if not os.path.exists(osm_path):
            raise FileNotFoundError(f"OSM data file not found: {osm_path}")
        osm_extractor = OSMFeatureExtractor(osm_path)

    csv_path = DEFAULT_INPUT
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    print(f"Loading data from: {csv_path}")
    df, stats = load_and_clean(csv_path, n_sample=n_sample, max_clean_samples=6000)
    print(f"Cleaned trajectories kept: {len(df)}")
    if len(df) < 4000:
        print("Warning: cleaned sample count is below 4000. The dataset slice may be sparse.")

    ratios = [0.2, 0.4, 0.6]
    results: Dict[float, Dict[str, Dict[str, float]]] = {}
    class_counts: Dict[float, int] = {}
    seq_results: Dict[float, Dict[str, float]] = {}
    seq_note = "- SeqTransformer was skipped because torch is unavailable."

    scatter_payload = None
    feature_importance_model = None
    feature_importance_names: List[str] = []
    best_ratio = ratios[0]
    best_method = "Ridge regression"
    best_mean = float("inf")

    def register_result(ratio: float, key: str, display_name: str, metrics: Dict[str, float]) -> None:
        nonlocal best_ratio, best_method, best_mean
        results.setdefault(ratio, {})[key] = metrics
        if metrics["mean_km"] < best_mean:
            best_mean = metrics["mean_km"]
            best_ratio = ratio
            best_method = display_name

    for ratio in ratios:
        feat_df = build_feature_frame(df, ratio, grid_cfg, osm_extractor=osm_extractor)
        feature_cols = [
            "start_lat",
            "start_lon",
            "end_lat",
            "end_lon",
            "prefix_dist",
            "prefix_duration",
            "prefix_avg_speed",
            "bearing",
            "bearing_std",
            "straight_line_dist",
            "detour_ratio",
            "hour_of_day",
            "weekday",
            "displacement_x",
            "displacement_y",
        ]
        if osm_extractor is not None:
            feature_cols.extend([
                "start_nearest_major_m",
                "start_nearest_secondary_m",
                "start_local_road_count_r200",
                "start_major_road_count_r300",
                "start_node_density_r200",
                "start_dominant_road_type_enc",
                "end_nearest_major_m",
                "end_nearest_secondary_m",
                "end_local_road_count_r200",
                "end_major_road_count_r300",
                "end_node_density_r200",
                "end_dominant_road_type_enc",
            ])
        X = feat_df[feature_cols].copy()
        y_reg = feat_df[["true_end_lat", "true_end_lon"]].values
        y_cls = feat_df["grid_id"].values
        class_counts[ratio] = int(pd.Series(y_cls).nunique())

        reg_model, x_test_reg, y_test_reg, pred_reg = split_and_train_regression(X, y_reg, seed=seed)
        cls_model, x_test_cls, y_test_cls, pred_cls_grid = split_and_train_classification(X, y_cls, seed=seed)

        _, x_test, _, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=seed)
        _, _, _, y_test_cls = train_test_split(X, y_cls, test_size=0.2, random_state=seed)
        pred_reg = reg_model.predict(x_test)
        pred_cls_grid = cls_model.predict(x_test)

        pred_cls_latlon = np.array([grid_id_to_center(int(g), grid_cfg) for g in pred_cls_grid])
        reg_errors = haversine_errors_km(y_test_reg, pred_reg)
        cls_errors = haversine_errors_km(y_test_reg, pred_cls_latlon)

        register_result(ratio, "regression", "Ridge regression", summarize_errors(reg_errors))
        register_result(ratio, "classification", "Grid classification", summarize_errors(cls_errors))

        if ratio == 0.6:
            scatter_payload = {
                "true": y_test_reg,
                "reg_pred": pred_reg,
                "cls_pred": pred_cls_latlon,
                "reg_errors": reg_errors,
                "cls_errors": cls_errors,
            }
            feature_importance_model = cls_model
            feature_importance_names = feature_cols

        print(
            f"Prefix {int(ratio * 100)}% | "
            f"reg mean={results[ratio]['regression']['mean_km']:.4f} km, "
            f"cls mean={results[ratio]['classification']['mean_km']:.4f} km"
        )

    assert scatter_payload is not None

    if TORCH_AVAILABLE:
        seq_ratio = 0.6
        seq_points = SEQ_MAX_LEN
        seq_df = df.sample(n=min(len(df), SEQ_SAMPLE_LIMIT), random_state=seed).reset_index(drop=True)
        print(f"SeqTransformer sample size: {len(seq_df)}")
        try:
            X_tok_60, X_meta_60, y_60 = build_seq_dataset(seq_df, seq_ratio, seq_max_len=seq_points, osm_extractor=osm_extractor)
            seq_model, seq_test_pred, seq_test_true, seq_metrics = fit_seq_transformer(
                X_tok_60,
                X_meta_60,
                y_60,
                seed=seed,
                epochs=SEQ_EPOCHS,
                batch_size=SEQ_BATCH_SIZE,
                patience=SEQ_PATIENCE,
            )
            if seq_metrics is not None:
                seq_results[0.6] = seq_metrics
                register_result(0.6, "seqtransformer", "SeqTransformer", seq_metrics)
                seq_note = f"- SeqTransformer trained on {seq_metrics['n_train']} train / {seq_metrics['n_val']} val / {seq_metrics['n_test']} test samples at 60% prefix."
                print(f"SeqTransformer | 60% mean={seq_metrics['mean_km']:.4f} km")

                if 0.4 not in seq_results:
                    X_tok_40, X_meta_40, y_40 = build_seq_dataset(seq_df, 0.4, seq_max_len=seq_points, osm_extractor=osm_extractor)
                    _, _, _, seq_metrics_40 = fit_seq_transformer(
                        X_tok_40,
                        X_meta_40,
                        y_40,
                        seed=seed,
                        epochs=max(16, SEQ_EPOCHS - 4),
                        batch_size=SEQ_BATCH_SIZE,
                        patience=SEQ_PATIENCE,
                    )
                    if seq_metrics_40 is not None:
                        seq_results[0.4] = seq_metrics_40
                        register_result(0.4, "seqtransformer", "SeqTransformer", seq_metrics_40)
                        print(f"SeqTransformer | 40% mean={seq_metrics_40['mean_km']:.4f} km")
        except Exception as exc:
            print(f"SeqTransformer skipped due to error: {exc}")
            seq_note = f"- SeqTransformer failed during training: {exc}"

        # Token variants use the same 60% sampled trajectories.
        try:
            train_idx, test_idx = train_test_split(np.arange(len(seq_df)), test_size=0.2, random_state=seed)
            train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=seed)
            train_df = seq_df.iloc[train_idx].reset_index(drop=True)

            grid_tokenizer = GridTokenizer(cell_lat=0.0018, cell_lon=0.0022)
            X_grid_tok, X_grid_aux, X_grid_meta, y_grid = build_grid_seq_dataset(seq_df, seq_ratio, grid_tokenizer, seq_max_len=seq_points, osm_extractor=osm_extractor)
            _, grid_pred, grid_true, grid_metrics = fit_token_transformer(
                X_grid_tok,
                X_grid_aux,
                X_grid_meta,
                y_grid,
                seed=seed,
                epochs=SEQ_EPOCHS,
                batch_size=SEQ_BATCH_SIZE,
                patience=SEQ_PATIENCE,
            )
            if grid_metrics is not None:
                register_result(0.6, "grid_token", "Grid Token Transformer", grid_metrics)
                print(f"GridTokenTransformer | 60% mean={grid_metrics['mean_km']:.4f} km")

            cluster_tokenizer = ClusterTokenizer(n_clusters=256, random_state=seed)
            cluster_tokenizer.fit(collect_prefix_points(train_df, seq_ratio))
            X_cluster_tok, X_cluster_aux, X_cluster_meta, y_cluster = build_cluster_seq_dataset(seq_df, seq_ratio, cluster_tokenizer, seq_max_len=seq_points, osm_extractor=osm_extractor)
            _, cluster_pred, cluster_true, cluster_metrics = fit_token_transformer(
                X_cluster_tok,
                X_cluster_aux,
                X_cluster_meta,
                y_cluster,
                seed=seed,
                epochs=SEQ_EPOCHS,
                batch_size=SEQ_BATCH_SIZE,
                patience=SEQ_PATIENCE,
            )
            if cluster_metrics is not None:
                register_result(0.6, "cluster_token", "Cluster Token Transformer", cluster_metrics)
                print(f"ClusterTokenTransformer | 60% mean={cluster_metrics['mean_km']:.4f} km")
        except Exception as exc:
            print(f"TokenTransformer variants skipped due to error: {exc}")
    else:
        print("SeqTransformer skipped because torch is unavailable.")

    line_plot = os.path.join(output_dir, "task3_prefix_error_line.png")
    scatter_plot = os.path.join(output_dir, "task3_destination_scatter.png")
    hist_plot = os.path.join(output_dir, "task3_error_histogram.png")
    fi_plot = os.path.join(output_dir, "task3_feature_importance.png")
    all_methods_bar_plot = os.path.join(output_dir, "task3_all_methods_60pct.png")

    save_line_plot(results, line_plot)
    save_scatter_plot(scatter_payload["true"], scatter_payload["reg_pred"], scatter_payload["cls_pred"], scatter_plot, title_suffix="(60% prefix)")
    save_histogram(scatter_payload["reg_errors"], scatter_payload["cls_errors"], seq_results.get(0.6) and np.asarray([seq_results[0.6]["mean_km"]]), hist_plot)
    save_feature_importance(feature_importance_model, feature_importance_names, fi_plot)
    save_all_methods_comparison_plot(results, all_methods_bar_plot)

    summary_path = write_summary(output_dir, stats, results, class_counts, best_ratio, best_method, seq_note)

    with open(os.path.join(output_dir, "task3_results.json"), "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "results": results, "best_ratio": best_ratio, "best_method": best_method, "seq_note": seq_note}, f, indent=2)

    print("\nDone.")
    print(f"Summary: {summary_path}")
    print(f"Plots: {line_plot}, {scatter_plot}, {hist_plot}, {fi_plot}, {all_methods_bar_plot}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Taxi destination prediction from trajectory prefixes")
    parser.add_argument("--n-sample", type=int, default=10000, help="Rows to read from train.csv")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory for figures and summary")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use-osm-features", action="store_true", default=False, help="Enable OSM semantic features")
    parser.add_argument("--osm-json", type=str, default='data/porto_highways_bbox.osm.json', help="OSM JSON path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(n_sample=args.n_sample, output_dir=os.path.expanduser(args.output_dir), seed=args.seed, osm_json=os.path.expanduser(args.osm_json) if args.use_osm_features else None)
