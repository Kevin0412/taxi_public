"""
出租车行程时长预测 —— Task 4：Travel Duration Prediction
数据集：Porto Taxi Trajectory（Kaggle）

本版本在原始课程作业脚本基础上做了两类增强：
1. 更严格的数据清洗：坐标范围、孤立飘点、相邻点超速检测
2. 更丰富的特征与实验：动态轨迹特征 + 轻量“序列式”实验
"""

import sys
import os
import argparse
import json
import math
import warnings
from collections import Counter, defaultdict

local_site = f"/home/test/.local/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
if os.path.isdir(local_site) and local_site not in sys.path:
    sys.path.insert(0, local_site)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PORTO_BBOX = {
    'lat_min': 41.05,
    'lat_max': 41.26,
    'lon_min': -8.75,
    'lon_max': -8.50,
}
SPEED_LIMIT_KMH = 120.0
GRID_SIZE = 0.01


# ══════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════
def haversine(lat1, lon1, lat2, lon2):
    """计算两点间球面距离（米）"""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing(lat1, lon1, lat2, lon2):
    """计算方位角（度）"""
    y = math.sin(math.radians(lon2 - lon1)) * math.cos(math.radians(lat2))
    x = (
        math.cos(math.radians(lat1)) * math.sin(math.radians(lat2))
        - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(lon2 - lon1))
    )
    deg = math.degrees(math.atan2(y, x))
    return (deg + 360.0) % 360.0


def angle_diff_deg(a, b):
    """最小方向夹角（0~180）"""
    return abs((a - b + 180) % 360 - 180)


def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_predictions(y_true, pred):
    pred = np.clip(np.asarray(pred), 0, None)
    return {
        'MAE': mean_absolute_error(y_true, pred),
        'RMSE': mean_squared_error(y_true, pred) ** 0.5,
        'MAPE': mape(y_true, pred),
        'pred': pred,
    }


def point_in_porto(lon, lat):
    return (
        PORTO_BBOX['lon_min'] <= lon <= PORTO_BBOX['lon_max']
        and PORTO_BBOX['lat_min'] <= lat <= PORTO_BBOX['lat_max']
    )


def lon_to_grid(lon):
    return int((lon - PORTO_BBOX['lon_min']) / GRID_SIZE)


def lat_to_grid(lat):
    return int((lat - PORTO_BBOX['lat_min']) / GRID_SIZE)


def segment_distances(points):
    return [
        haversine(points[i - 1][1], points[i - 1][0], points[i][1], points[i][0])
        for i in range(1, len(points))
    ]


def segment_speeds(points, delta_seconds=15.0):
    return [d / delta_seconds for d in segment_distances(points)]


def segment_bearings(points):
    return [
        bearing(points[i - 1][1], points[i - 1][0], points[i][1], points[i][0])
        for i in range(1, len(points))
    ]


def parse_polyline(poly_str):
    try:
        pts = json.loads(poly_str)
    except Exception:
        return None

    if not isinstance(pts, list):
        return None

    clean = []
    for p in pts:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        try:
            lon, lat = float(p[0]), float(p[1])
        except Exception:
            continue
        if math.isnan(lon) or math.isnan(lat):
            continue
        if point_in_porto(lon, lat):
            clean.append([lon, lat])

    return clean if len(clean) >= 2 else None


def lonlat_to_local_xy(lon, lat, ref_lat=41.157):
    x = math.radians(lon) * 6371000 * math.cos(math.radians(ref_lat))
    y = math.radians(lat) * 6371000
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


# ══════════════════════════════════════════════════════════════════════
# 1. 数据生成（合成数据，结构与 Porto 数据集兼容）
# ══════════════════════════════════════════════════════════════════════
def set_global_seed(seed=42):
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass

    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════
# 1. 数据生成（合成数据，结构与 Porto 数据集兼容）
# ══════════════════════════════════════════════════════════════════════
def generate_synthetic_data(n=5000, seed=42):
    print("生成合成数据（模拟 Porto Taxi 轨迹）...")
    set_global_seed(seed)

    records = []
    center_lat, center_lon = 41.157, -8.629

    for i in range(n):
        timestamp = int(1376503200 + np.random.randint(0, 30 * 24 * 3600))
        hour = (timestamp // 3600) % 24
        weekday = (timestamp // 86400) % 7

        start_lat = center_lat + np.random.uniform(-0.05, 0.05)
        start_lon = center_lon + np.random.uniform(-0.08, 0.08)

        base_steps = np.random.randint(10, 80)
        if hour in [8, 9, 17, 18, 19]:
            base_steps = int(base_steps * 1.4)

        duration_true = base_steps * 15

        polyline = [[start_lon, start_lat]]
        lat, lon = start_lat, start_lon
        for _ in range(base_steps - 1):
            lat += np.random.uniform(-0.002, 0.002)
            lon += np.random.uniform(-0.003, 0.003)
            lat = min(max(lat, PORTO_BBOX['lat_min'] + 0.001), PORTO_BBOX['lat_max'] - 0.001)
            lon = min(max(lon, PORTO_BBOX['lon_min'] + 0.001), PORTO_BBOX['lon_max'] - 0.001)
            polyline.append([round(lon, 6), round(lat, 6)])

        prefix_len = max(2, int(len(polyline) * 0.4))
        prefix = polyline[:prefix_len]

        records.append({
            'TRIP_ID': i,
            'TIMESTAMP': timestamp,
            'TAXI_ID': np.random.randint(1, 100),
            'CALL_TYPE': np.random.choice(['A', 'B', 'C'], p=[0.3, 0.2, 0.5]),
            'DAYTYPE': 'A',
            'MISSING_DATA': False,
            'POLYLINE': json.dumps(polyline),
            'DURATION_TRUE': duration_true,
            '_prefix': json.dumps(prefix),
            '_hour': hour,
            '_weekday': weekday,
            '_start_lat': start_lat,
            '_start_lon': start_lon,
        })

    df = pd.DataFrame(records)
    print(f"合成数据：{len(df)} 条，行程时长范围：{df['DURATION_TRUE'].min()}s ~ {df['DURATION_TRUE'].max()}s")
    return df


# ══════════════════════════════════════════════════════════════════════
# 2. 真实数据清洗（Porto 格式）
# ══════════════════════════════════════════════════════════════════════
def remove_isolated_spikes(points, speed_limit_kmh=SPEED_LIMIT_KMH):
    """删除典型孤立飘点：前后都跳很远，但跳过该点后恢复正常。"""
    if len(points) < 3:
        return points, 0

    speed_limit = speed_limit_kmh / 3.6
    cleaned = [points[0]]
    removed = 0
    i = 1

    while i < len(points) - 1:
        prev_pt = cleaned[-1]
        curr_pt = points[i]
        next_pt = points[i + 1]

        d1 = haversine(prev_pt[1], prev_pt[0], curr_pt[1], curr_pt[0])
        d2 = haversine(curr_pt[1], curr_pt[0], next_pt[1], next_pt[0])
        d_skip = haversine(prev_pt[1], prev_pt[0], next_pt[1], next_pt[0])

        v1 = d1 / 15.0
        v2 = d2 / 15.0
        v_skip = d_skip / 30.0

        looks_like_spike = (
            (v1 > speed_limit or v2 > speed_limit)
            and v_skip <= speed_limit
            and d_skip < max(d1, d2)
        )

        if looks_like_spike:
            removed += 1
            i += 1
            continue

        cleaned.append(curr_pt)
        i += 1

    cleaned.append(points[-1])
    return cleaned, removed


def clean_real_trajectory(poly_str, speed_limit_kmh=SPEED_LIMIT_KMH):
    stats = {
        'reason': None,
        'removed_spikes': 0,
        'bad_segments': 0,
        'max_speed_kmh': 0.0,
    }

    points = parse_polyline(poly_str)
    if points is None:
        stats['reason'] = 'parse_or_bbox'
        return None, stats

    points, removed = remove_isolated_spikes(points, speed_limit_kmh=speed_limit_kmh)
    stats['removed_spikes'] = removed

    if len(points) < 2:
        stats['reason'] = 'too_short_after_clean'
        return None, stats

    speeds = segment_speeds(points)
    if not speeds:
        stats['reason'] = 'too_short_after_clean'
        return None, stats

    speed_limit = speed_limit_kmh / 3.6
    bad_segments = sum(v > speed_limit for v in speeds)
    stats['bad_segments'] = bad_segments
    stats['max_speed_kmh'] = max(speeds) * 3.6

    if stats['max_speed_kmh'] > 180:
        stats['reason'] = 'extreme_speed'
        return None, stats

    if bad_segments > 2 or bad_segments / max(1, len(speeds)) > 0.15:
        stats['reason'] = 'too_many_fast_segments'
        return None, stats

    return points, stats


def load_real_data(csv_path, n_sample=50000, speed_limit_kmh=SPEED_LIMIT_KMH):
    print(f"加载真实数据：{csv_path}")
    df = pd.read_csv(csv_path, nrows=n_sample)
    print(f"原始读取：{len(df)} 条")

    stats = Counter()
    stats['raw_rows'] = len(df)

    df = df[df['MISSING_DATA'] == False].copy()
    stats['after_missing_filter'] = len(df)

    records = []
    total_removed_spikes = 0

    for _, row in df.iterrows():
        points, info = clean_real_trajectory(row['POLYLINE'], speed_limit_kmh=speed_limit_kmh)
        if points is None:
            stats[info['reason']] += 1
            total_removed_spikes += info.get('removed_spikes', 0)
            continue

        duration_true = (len(points) - 1) * 15
        if not (60 <= duration_true <= 7200):
            stats['duration_out_of_range'] += 1
            total_removed_spikes += info.get('removed_spikes', 0)
            continue

        rec = row.to_dict()
        rec['_poly_clean'] = points
        rec['DURATION_TRUE'] = duration_true
        rec['_clean_removed_spikes'] = info['removed_spikes']
        rec['_clean_bad_segments'] = info['bad_segments']
        rec['_clean_max_speed_kmh'] = info['max_speed_kmh']
        records.append(rec)
        total_removed_spikes += info.get('removed_spikes', 0)

    clean_df = pd.DataFrame(records)
    stats['after_clean'] = len(clean_df)
    stats['removed_spikes_total'] = total_removed_spikes

    print(f"过滤缺失后：{stats['after_missing_filter']} 条")
    print(f"清洗后有效数据：{stats['after_clean']} 条")
    print(f"累计移除孤立飘点：{stats['removed_spikes_total']} 个")

    drop_keys = [
        'parse_or_bbox',
        'too_short_after_clean',
        'extreme_speed',
        'too_many_fast_segments',
        'duration_out_of_range',
    ]
    for k in drop_keys:
        if stats.get(k, 0):
            print(f"  丢弃 {k:24s}: {stats[k]}")

    return clean_df, dict(stats)


# ══════════════════════════════════════════════════════════════════════
# 3. 特征工程（增强版）
# ══════════════════════════════════════════════════════════════════════
def get_prefix_and_context(row, df_columns, prefix_ratio=None, prefix_points=10):
    if '_prefix' in df_columns:
        prefix = json.loads(row['_prefix'])
        hour = row['_hour']
        weekday = row['_weekday']
        start_lat = row['_start_lat']
        start_lon = row['_start_lon']
        poly = None
    else:
        poly = row['_poly_clean'] if '_poly_clean' in row else json.loads(row['POLYLINE'])
        ts = int(row['TIMESTAMP'])
        hour = (ts // 3600) % 24
        weekday = (ts // 86400) % 7
        start_lon, start_lat = poly[0]

        if prefix_points is not None:
            if len(poly) <= prefix_points:
                return None
            prefix = poly[:prefix_points]
        else:
            prefix_n = max(2, int(len(poly) * prefix_ratio))
            if prefix_n >= len(poly):
                return None
            prefix = poly[:prefix_n]

    return {
        'poly': poly,
        'prefix': prefix,
        'hour': hour,
        'weekday': weekday,
        'start_lat': start_lat,
        'start_lon': start_lon,
    }


def extract_features(df, prefix_ratio=None, prefix_points=10, osm_extractor=None):
    print("\n提取增强特征...")

    rows = []
    df_columns = set(df.columns)

    for _, row in df.iterrows():
        try:
            ctx = get_prefix_and_context(row, df_columns, prefix_ratio=prefix_ratio, prefix_points=prefix_points)
            if ctx is None:
                continue

            prefix = ctx['prefix']
            if len(prefix) < 2:
                continue

            hour = ctx['hour']
            weekday = ctx['weekday']
            start_lat = ctx['start_lat']
            start_lon = ctx['start_lon']

            p_end_lon, p_end_lat = prefix[-1]
            dists = segment_distances(prefix)
            speeds = [d / 15.0 for d in dists]
            bearings = segment_bearings(prefix)
            turns = [angle_diff_deg(bearings[i], bearings[i - 1]) for i in range(1, len(bearings))]
            accels = [abs(speeds[i] - speeds[i - 1]) / 15.0 for i in range(1, len(speeds))]

            dist_total = float(np.sum(dists)) if dists else 0.0
            straight = haversine(prefix[0][1], prefix[0][0], p_end_lat, p_end_lon)
            avg_speed = float(np.mean(speeds)) if speeds else 0.0
            speed_std = float(np.std(speeds)) if len(speeds) > 1 else 0.0
            speed_max = float(np.max(speeds)) if speeds else 0.0
            speed_min = float(np.min(speeds)) if speeds else 0.0
            br = bearing(prefix[0][1], prefix[0][0], p_end_lat, p_end_lon)
            turn_mean = float(np.mean(turns)) if turns else 0.0
            turn_count = int(sum(t > 30 for t in turns))
            accel_std = float(np.std(accels)) if len(accels) > 1 else 0.0
            detour_ratio = dist_total / max(straight, 1.0)
            recent_speed_mean_3 = float(np.mean(speeds[-3:])) if speeds else 0.0
            recent_speed_mean_5 = float(np.mean(speeds[-5:])) if speeds else 0.0
            recent_turn_mean_3 = float(np.mean(turns[-3:])) if turns else 0.0
            zero_move_ratio = float(np.mean(np.array(dists) < 5.0)) if dists else 0.0

            call_enc = {'A': 0, 'B': 1, 'C': 2}.get(str(row.get('CALL_TYPE', 'C')), 2)
            osm_features = {}
            if osm_extractor is not None:
                osm_features.update(osm_extractor.point_features(start_lon, start_lat, 'start'))
                osm_features.update(osm_extractor.point_features(p_end_lon, p_end_lat, 'end'))

            rows.append({
                'hour': hour,
                'weekday': weekday,
                'is_peak': int(hour in [8, 9, 17, 18, 19]),
                'start_lat': start_lat,
                'start_lon': start_lon,
                'prefix_end_lat': p_end_lat,
                'prefix_end_lon': p_end_lon,
                'prefix_dist_m': dist_total,
                'prefix_speed': avg_speed,
                'prefix_speed_std': speed_std,
                'prefix_speed_max': speed_max,
                'prefix_speed_min': speed_min,
                'prefix_straight': straight,
                'prefix_bearing': br,
                'prefix_len_pts': len(prefix),
                'prefix_turn_mean': turn_mean,
                'prefix_turn_count': turn_count,
                'prefix_accel_std': accel_std,
                'prefix_detour_ratio': detour_ratio,
                'recent_speed_mean_3': recent_speed_mean_3,
                'recent_speed_mean_5': recent_speed_mean_5,
                'recent_turn_mean_3': recent_turn_mean_3,
                'zero_move_ratio': zero_move_ratio,
                'prefix_ratio_traveled': min(1.0, straight / 5000.0),
                'call_type_enc': call_enc,
                'start_grid_x': lon_to_grid(start_lon),
                'start_grid_y': lat_to_grid(start_lat),
                'end_grid_x': lon_to_grid(p_end_lon),
                'end_grid_y': lat_to_grid(p_end_lat),
                **osm_features,
                'duration': row['DURATION_TRUE'],
            })
        except Exception:
            continue

    feat_df = pd.DataFrame(rows)
    print(f"特征矩阵：{feat_df.shape[0]} 行 × {max(0, feat_df.shape[1] - 1)} 特征")
    return feat_df


# ══════════════════════════════════════════════════════════════════════
# 4. 建模与评估（增强版表格模型）
# ══════════════════════════════════════════════════════════════════════
def train_and_evaluate(feat_df):
    print("\n" + "=" * 60)
    print("增强版模型训练与评估")
    print("=" * 60)

    feat_cols = [c for c in feat_df.columns if c != 'duration']
    X = feat_df[feat_cols].fillna(0).values
    y = feat_df['duration'].values

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    baseline_pred = np.full_like(y_test, y_train.mean(), dtype=float)
    baseline_res = evaluate_predictions(y_test, baseline_pred)
    bl_mae = baseline_res['MAE']
    print(
        f"\n  Baseline（均值）:  MAE={baseline_res['MAE']:.1f}s  "
        f"RMSE={baseline_res['RMSE']:.1f}s  MAPE={baseline_res['MAPE']:.1f}%"
    )

    models = {
        'LinearRegression': (LinearRegression(), True, False),
        'Ridge(α=10)': (Ridge(alpha=10), True, False),
        'RandomForest': (
            RandomForestRegressor(n_estimators=140, max_depth=14, random_state=42, n_jobs=-1),
            False,
            False,
        ),
        'GradientBoosting': (
            GradientBoostingRegressor(n_estimators=240, max_depth=4, learning_rate=0.05, random_state=42),
            False,
            False,
        ),
        'HistGradientBoosting': (
            HistGradientBoostingRegressor(max_depth=8, learning_rate=0.05, max_iter=260, random_state=42),
            False,
            False,
        ),
        'HistGB(log1p)': (
            HistGradientBoostingRegressor(max_depth=8, learning_rate=0.05, max_iter=260, random_state=42),
            False,
            True,
        ),
    }

    results = {
        'Baseline': {
            'model': None,
            'pred': baseline_res['pred'],
            'MAE': baseline_res['MAE'],
            'RMSE': baseline_res['RMSE'],
            'MAPE': baseline_res['MAPE'],
            'improvement': 0.0,
            'scaled': False,
            'log_target': False,
        }
    }

    for name, (model, use_scaled, log_target) in models.items():
        Xtr = X_train_s if use_scaled else X_train
        Xte = X_test_s if use_scaled else X_test
        ytr = np.log1p(y_train) if log_target else y_train

        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        if log_target:
            pred = np.expm1(pred)

        metrics = evaluate_predictions(y_test, pred)
        impv = (bl_mae - metrics['MAE']) / bl_mae * 100
        results[name] = {
            'model': model,
            'pred': metrics['pred'],
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'MAPE': metrics['MAPE'],
            'improvement': impv,
            'scaled': use_scaled,
            'log_target': log_target,
        }
        print(
            f"  {name:22s}: MAE={metrics['MAE']:.1f}s  RMSE={metrics['RMSE']:.1f}s  "
            f"MAPE={metrics['MAPE']:.1f}%  提升={impv:.1f}%"
        )

    best_name = min([k for k in results if k != 'Baseline'], key=lambda k: results[k]['MAE'])
    best = results[best_name]
    print(f"\n  最优模型：{best_name}  MAE={best['MAE']:.1f}s")

    return results, best_name, feat_cols, scaler


# ══════════════════════════════════════════════════════════════════════
# 5. 前缀长度分析
# ══════════════════════════════════════════════════════════════════════
def prefix_analysis(df_raw, osm_extractor=None):
    print("\n" + "=" * 60)
    print("观察窗口分析（5 / 10 / 20 个轨迹点）")
    print("=" * 60)

    prefix_results = {}
    for pts in [5, 10, 20]:
        feat_df = extract_features(df_raw, prefix_points=pts, osm_extractor=osm_extractor)
        if len(feat_df) < 100:
            continue

        X = feat_df[[c for c in feat_df.columns if c != 'duration']].fillna(0).values
        y = feat_df['duration'].values
        split = int(len(X) * 0.8)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        rf = RandomForestRegressor(n_estimators=120, max_depth=14, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        pred = rf.predict(X_te)
        mae_ = mean_absolute_error(y_te, pred)
        prefix_results[pts] = mae_
        print(f"  前缀 {pts:2d} 点: MAE={mae_:.1f}s")

    return prefix_results


# ══════════════════════════════════════════════════════════════════════
# 6. 误差分析
# ══════════════════════════════════════════════════════════════════════
def error_analysis(feat_df, results, best_name):
    print("\n" + "=" * 60)
    print("误差分析")
    print("=" * 60)

    split = int(len(feat_df) * 0.8)
    test_df = feat_df.iloc[split:].copy().reset_index(drop=True)
    test_df['pred'] = results[best_name]['pred']
    test_df['error'] = test_df['pred'] - test_df['duration']
    test_df['abs_error'] = test_df['error'].abs()

    print("\n【类型 1】高峰时段误差（高峰 vs 非高峰）：")
    for grp, sub in test_df.groupby('is_peak'):
        label = '高峰' if grp == 1 else '非高峰'
        print(
            f"  {label}: 平均误差={sub['abs_error'].mean():.1f}s  "
            f"中位误差={sub['abs_error'].median():.1f}s  样本数={len(sub)}"
        )

    print("\n【类型 2】长短途误差（按真实时长分组）：")
    test_df['trip_cat'] = pd.cut(
        test_df['duration'],
        bins=[0, 300, 600, 1200, 99999],
        labels=['<5min', '5-10min', '10-20min', '>20min'],
    )
    for cat, sub in test_df.groupby('trip_cat', observed=True):
        print(f"  {cat}: 平均误差={sub['abs_error'].mean():.1f}s  样本数={len(sub)}")

    print("\n【类型 3】低速行驶（堵车）误差：")
    test_df['speed_cat'] = pd.cut(
        test_df['prefix_speed'],
        bins=[0, 3, 8, 100],
        labels=['低速(<3m/s)', '中速(3-8m/s)', '高速(>8m/s)'],
    )
    for cat, sub in test_df.groupby('speed_cat', observed=True):
        print(f"  {cat}: 平均误差={sub['abs_error'].mean():.1f}s  样本数={len(sub)}")

    print("\n【类型 4】曲折度分组误差：")
    test_df['detour_cat'] = pd.cut(
        test_df['prefix_detour_ratio'],
        bins=[0, 1.2, 1.8, 3.0, 999],
        labels=['接近直线', '中等弯折', '较绕路', '极绕'],
    )
    for cat, sub in test_df.groupby('detour_cat', observed=True):
        print(f"  {cat}: 平均误差={sub['abs_error'].mean():.1f}s  样本数={len(sub)}")

    print("\n误差最大的 5 条记录：")
    top5 = test_df.nlargest(5, 'abs_error')[
        ['duration', 'pred', 'abs_error', 'hour', 'is_peak', 'prefix_speed', 'prefix_detour_ratio']
    ]
    print(top5.round(1).to_string(index=False))

    return test_df


# ══════════════════════════════════════════════════════════════════════
# 7. 可视化
# ══════════════════════════════════════════════════════════════════════
def visualize_all(feat_df, results, best_name, prefix_results, test_err_df, seq_results=None):
    print("\n生成可视化图表...")

    pred = results[best_name]['pred']
    true = test_err_df['duration'].values

    # 图1：预测 vs 真实
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sample = np.random.choice(len(pred), min(1000, len(pred)), replace=False)
    axes[0].scatter(true[sample], pred[sample], alpha=0.4, s=12, color='steelblue')
    lo, hi = 0, np.percentile(true, 99)
    axes[0].plot([lo, hi], [lo, hi], 'r--', linewidth=1.5)
    axes[0].set_xlabel('True Duration (s)')
    axes[0].set_ylabel('Predicted Duration (s)')
    axes[0].set_title(f'{best_name}: True vs Predicted')

    res = pred - true
    axes[1].hist(res, bins=80, color='coral', edgecolor='white', alpha=0.8)
    axes[1].axvline(0, color='red', linestyle='--')
    axes[1].set_xlabel('Residual (s)')
    axes[1].set_title('Residual Distribution')
    axes[1].text(
        0.97,
        0.97,
        f"MAE={results[best_name]['MAE']:.0f}s\nRMSE={results[best_name]['RMSE']:.0f}s\nMAPE={results[best_name]['MAPE']:.1f}%",
        transform=axes[1].transAxes,
        ha='right',
        va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'taxi_01_prediction.png'), dpi=150)
    plt.close()

    # 图2：模型对比
    model_names = [k for k in results.keys() if k != 'Baseline']
    maes = [results[k]['MAE'] for k in model_names]
    rmses = [results[k]['RMSE'] for k in model_names]
    mapes = [results[k]['MAPE'] for k in model_names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, vals, title, col in [
        (axes[0], maes, 'MAE (s)', 'steelblue'),
        (axes[1], rmses, 'RMSE (s)', 'salmon'),
        (axes[2], mapes, 'MAPE (%)', 'seagreen'),
    ]:
        x = np.arange(len(model_names))
        ax.bar(x, vals, color=col, edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=20, ha='right')
        ax.set_title(title)
        for i, v in enumerate(vals):
            ax.text(i, v * 1.01, f'{v:.1f}', ha='center', fontsize=9)
    plt.suptitle('Enhanced Model Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'taxi_02_model_comparison.png'), dpi=150)
    plt.close()

    # 图3：前缀长度 vs MAE
    if prefix_results:
        obs_points = sorted(prefix_results.keys())
        maes_p = [prefix_results[p] for p in obs_points]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(obs_points, maes_p, 'o-', color='steelblue', linewidth=2, markersize=8)
        for p, m in zip(obs_points, maes_p):
            ax.text(p, m + 5, f'{m:.0f}s', ha='center', fontsize=10)
        ax.set_xlabel('Observed Prefix Points')
        ax.set_ylabel('MAE (s)')
        ax.set_title('MAE vs Observed Prefix Length (Random Forest)')
        ax.set_xticks(obs_points)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'taxi_03_prefix_analysis.png'), dpi=150)
        plt.close()

    # 图4：误差分析
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cat_err = test_err_df.groupby('trip_cat', observed=True)['abs_error'].mean()
    axes[0].bar(cat_err.index.astype(str), cat_err.values, color='steelblue', edgecolor='white')
    axes[0].set_xlabel('Trip Duration Category')
    axes[0].set_ylabel('Mean Absolute Error (s)')
    axes[0].set_title('Error by Trip Length')

    hour_err = test_err_df.groupby('hour')['abs_error'].mean()
    axes[1].bar(hour_err.index, hour_err.values, color='coral', edgecolor='white', alpha=0.8)
    axes[1].set_xlabel('Hour of Day')
    axes[1].set_ylabel('Mean Absolute Error (s)')
    axes[1].set_title('Error by Hour of Day')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'taxi_04_error_analysis.png'), dpi=150)
    plt.close()

    # 图5：特征重要性（随机森林）
    if 'RandomForest' in results and results['RandomForest']['model'] is not None:
        rf = results['RandomForest']['model']
        feat_cols = [c for c in feat_df.columns if c != 'duration']
        imp = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(9, 7))
        imp.tail(18).plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title('Feature Importance (Random Forest)')
        ax.set_xlabel('Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'taxi_05_feature_importance.png'), dpi=150)
        plt.close()

    # 图6：序列实验结果
    if seq_results:
        seq_names = list(seq_results.keys())
        seq_maes = [seq_results[k]['MAE'] for k in seq_names]
        seq_colors = ['#6baed6', '#fd8d3c', '#74c476', '#9e9ac8'][:len(seq_names)]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(seq_names, seq_maes, color=seq_colors, edgecolor='white')
        for i, v in enumerate(seq_maes):
            ax.text(i, v * 1.01, f'{v:.1f}', ha='center')
        ax.set_ylabel('MAE (s)')
        ax.set_title('Sequence-style Experiment Comparison')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'taxi_06_sequence_experiment.png'), dpi=150)
        plt.close()

    print(f"图表保存至 {OUTPUT_DIR}/")


# ══════════════════════════════════════════════════════════════════════
# 8. 轻量“序列式”实验：固定前缀展平 + MLP / HistGB + Transformer
# ══════════════════════════════════════════════════════════════════════
def build_sequence_dataset(df_raw, prefix_points=10):
    rows = []
    y = []

    for _, row in df_raw.iterrows():
        try:
            poly = row['_poly_clean'] if '_poly_clean' in row else json.loads(row['POLYLINE'])
            if len(poly) <= prefix_points:
                continue

            prefix = poly[:prefix_points]
            start_lon, start_lat = prefix[0]
            ts = int(row['TIMESTAMP'])
            hour = (ts // 3600) % 24
            weekday = (ts // 86400) % 7
            call_enc = {'A': 0, 'B': 1, 'C': 2}.get(str(row.get('CALL_TYPE', 'C')), 2)

            seq_feat = []
            for lon, lat in prefix:
                seq_feat.extend([lon - start_lon, lat - start_lat])

            for i in range(1, len(prefix)):
                lon0, lat0 = prefix[i - 1]
                lon1, lat1 = prefix[i]
                d = haversine(lat0, lon0, lat1, lon1)
                b = bearing(lat0, lon0, lat1, lon1)
                seq_feat.extend([
                    lon1 - lon0,
                    lat1 - lat0,
                    d / 100.0,
                    math.sin(math.radians(b)),
                    math.cos(math.radians(b)),
                ])

            seq_feat.extend([
                hour / 23.0,
                weekday / 6.0,
                int(hour in [8, 9, 17, 18, 19]),
                call_enc / 2.0,
            ])

            rows.append(seq_feat)
            y.append(row['DURATION_TRUE'])
        except Exception:
            continue

    return np.asarray(rows, dtype=float), np.asarray(y, dtype=float)


def build_transformer_sequence_dataset(df_raw, prefix_points=30):
    token_rows = []
    meta_rows = []
    y = []

    for _, row in df_raw.iterrows():
        try:
            poly = row['_poly_clean'] if '_poly_clean' in row else json.loads(row['POLYLINE'])
            if len(poly) <= prefix_points:
                continue

            prefix = poly[:prefix_points]
            start_lon, start_lat = prefix[0]
            ts = int(row['TIMESTAMP'])
            hour = (ts // 3600) % 24
            weekday = (ts // 86400) % 7
            is_peak = int(hour in [8, 9, 17, 18, 19])
            call_enc = {'A': 0, 'B': 1, 'C': 2}.get(str(row.get('CALL_TYPE', 'C')), 2)

            dists_m = []
            bearings_deg = []
            for i in range(1, len(prefix)):
                lon0, lat0 = prefix[i - 1]
                lon1, lat1 = prefix[i]
                dists_m.append(haversine(lat0, lon0, lat1, lon1))
                bearings_deg.append(bearing(lat0, lon0, lat1, lon1))

            total_dist_m = sum(dists_m)
            cumulative_dist_m = 0.0
            prev_lon, prev_lat = start_lon, start_lat
            prev_bearing = 0.0
            tokens = []
            for i, (lon, lat) in enumerate(prefix):
                rel_lon = lon - start_lon
                rel_lat = lat - start_lat
                rel_progress = i / max(prefix_points - 1, 1)
                if i == 0:
                    delta_lon = 0.0
                    delta_lat = 0.0
                    seg_dist_m = 0.0
                    speed_mps = 0.0
                    local_turn = 0.0
                    b = 0.0
                else:
                    delta_lon = lon - prev_lon
                    delta_lat = lat - prev_lat
                    seg_dist_m = dists_m[i - 1]
                    speed_mps = seg_dist_m / 15.0
                    b = bearings_deg[i - 1]
                    turn = ((b - prev_bearing + 180.0) % 360.0) - 180.0
                    local_turn = 0.0 if i == 1 else turn / 180.0
                    cumulative_dist_m += seg_dist_m
                straight_to_start_m = haversine(start_lat, start_lon, lat, lon)
                dist_share = cumulative_dist_m / max(total_dist_m, 1.0)
                tokens.append([
                    rel_lon,
                    rel_lat,
                    delta_lon,
                    delta_lat,
                    seg_dist_m / 100.0,
                    cumulative_dist_m / 1000.0,
                    straight_to_start_m / 1000.0,
                    speed_mps / 15.0,
                    local_turn,
                    dist_share,
                    rel_progress,
                    math.sin(math.radians(b)),
                    math.cos(math.radians(b)),
                ])
                prev_lon, prev_lat = lon, lat
                prev_bearing = b

            end_lon, end_lat = prefix[-1]
            straight_prefix_m = haversine(start_lat, start_lon, end_lat, end_lon)
            detour_ratio = total_dist_m / max(straight_prefix_m, 1.0)
            recent_mean_speed = np.mean(dists_m[-5:]) / 15.0 if dists_m else 0.0
            recent_turn_mean = 0.0
            if len(bearings_deg) >= 2:
                turns = [abs(((bearings_deg[j] - bearings_deg[j - 1] + 180.0) % 360.0) - 180.0) for j in range(1, len(bearings_deg))]
                recent_turn_mean = float(np.mean(turns[-5:])) / 180.0 if turns else 0.0

            token_rows.append(tokens)
            meta_rows.append([
                hour / 23.0,
                weekday / 6.0,
                is_peak,
                call_enc / 2.0,
                prefix_points / 40.0,
                total_dist_m / 1000.0,
                straight_prefix_m / 1000.0,
                detour_ratio,
                recent_mean_speed / 15.0,
                recent_turn_mean,
            ])
            y.append(row['DURATION_TRUE'])
        except Exception:
            continue

    return (
        np.asarray(token_rows, dtype=np.float32),
        np.asarray(meta_rows, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
    )


class SequenceTransformerRegressor(nn.Module if nn is not None else object):
    def __init__(self, token_dim, meta_dim, max_len, d_model=64, nhead=4, num_layers=2, dropout=0.12):
        super().__init__()
        self.token_proj = nn.Linear(token_dim, d_model)
        self.meta_proj = nn.Sequential(
            nn.Linear(meta_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 3,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.pool_score = nn.Linear(d_model, 1)
        self.head = nn.Sequential(
            nn.Linear(d_model * 3, 96),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(96, 1),
        )

    def forward(self, tokens, meta):
        tok = self.token_proj(tokens)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        x = torch.cat([cls, tok], dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.encoder(x)
        cls_feat = self.norm(x[:, 0, :])
        tok_feat = self.norm(x[:, 1:, :])
        attn = torch.softmax(self.pool_score(tok_feat), dim=1)
        attn_pool = torch.sum(tok_feat * attn, dim=1)
        meta_feat = self.meta_proj(meta)
        out = self.head(torch.cat([cls_feat, attn_pool, meta_feat], dim=1))
        return out.squeeze(1)


def fit_seq_transformer(X_tokens, X_meta, y, max_train_samples=24000, epochs=18, batch_size=384, patience=4):
    if not TORCH_AVAILABLE:
        print("  未检测到 torch，跳过 SeqTransformer。")
        return None

    if len(X_tokens) < 400:
        print("  序列样本过少，跳过 SeqTransformer。")
        return None

    set_global_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  SeqTransformer 设备：{device}")

    if len(X_tokens) > max_train_samples:
        idx = np.linspace(0, len(X_tokens) - 1, max_train_samples, dtype=int)
        X_tokens = X_tokens[idx]
        X_meta = X_meta[idx]
        y = y[idx]
        print(f"  SeqTransformer 训练样本上限：截取为 {len(X_tokens)} 条")

    idx = np.arange(len(X_tokens))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)
    Xtr_tok, Xte_tok = X_tokens[train_idx], X_tokens[test_idx]
    Xtr_meta, Xte_meta = X_meta[train_idx], X_meta[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if len(Xte_tok) < 50 or len(Xtr_tok) < 200:
        print("  SeqTransformer 可用训练/测试样本不足，跳过。")
        return None

    train_sub_idx, val_idx = train_test_split(np.arange(len(Xtr_tok)), test_size=0.1, random_state=42)
    if len(val_idx) < 32 or len(train_sub_idx) < 128:
        print("  SeqTransformer 验证集划分失败，跳过。")
        return None

    Xva_tok = Xtr_tok[val_idx]
    Xva_meta = Xtr_meta[val_idx]
    y_val = y_train[val_idx]
    Xtr_tok = Xtr_tok[train_sub_idx]
    Xtr_meta = Xtr_meta[train_sub_idx]
    y_train = y_train[train_sub_idx]

    token_mean = Xtr_tok.mean(axis=(0, 1), keepdims=True)
    token_std = Xtr_tok.std(axis=(0, 1), keepdims=True) + 1e-6
    meta_mean = Xtr_meta.mean(axis=0, keepdims=True)
    meta_std = Xtr_meta.std(axis=0, keepdims=True) + 1e-6

    Xtr_tok = (Xtr_tok - token_mean) / token_std
    Xva_tok = (Xva_tok - token_mean) / token_std
    Xte_tok = (Xte_tok - token_mean) / token_std
    Xtr_meta = (Xtr_meta - meta_mean) / meta_std
    Xva_meta = (Xva_meta - meta_mean) / meta_std
    Xte_meta = (Xte_meta - meta_mean) / meta_std

    ytr_log = np.log1p(y_train)
    yva_log = np.log1p(y_val)

    train_ds = TensorDataset(
        torch.tensor(Xtr_tok, dtype=torch.float32),
        torch.tensor(Xtr_meta, dtype=torch.float32),
        torch.tensor(ytr_log, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(Xva_tok, dtype=torch.float32),
        torch.tensor(Xva_meta, dtype=torch.float32),
        torch.tensor(yva_log, dtype=torch.float32),
    )
    test_tok_t = torch.tensor(Xte_tok, dtype=torch.float32)
    test_meta_t = torch.tensor(Xte_meta, dtype=torch.float32)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SequenceTransformerRegressor(
        token_dim=Xtr_tok.shape[2],
        meta_dim=Xtr_meta.shape[1],
        max_len=Xtr_tok.shape[1],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=1)
    criterion = nn.SmoothL1Loss(beta=0.25)

    best_state = None
    best_val = float('inf')
    wait = 0

    print(f"  SeqTransformer 开始训练：train={len(train_ds)} val={len(val_ds)} test={len(Xte_tok)}")
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch_tok, batch_meta, batch_y in train_loader:
            batch_tok = batch_tok.to(device)
            batch_meta = batch_meta.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_tok, batch_meta)
            loss = criterion(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_tok, batch_meta, batch_y in val_loader:
                batch_tok = batch_tok.to(device)
                batch_meta = batch_meta.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_tok, batch_meta)
                val_losses.append(criterion(pred, batch_y).item())

        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        val_loss = float(np.mean(val_losses)) if val_losses else np.nan
        lr_now = optimizer.param_groups[0]['lr']
        print(f"    epoch {epoch:02d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr={lr_now:.5f}")
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("  SeqTransformer 触发 early stopping。")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_log = model(test_tok_t.to(device), test_meta_t.to(device)).cpu().numpy()
    pred = np.expm1(pred_log)

    metrics = evaluate_predictions(y_test, pred)
    metrics['n_train'] = len(train_ds)
    metrics['n_val'] = len(val_ds)
    metrics['n_test'] = len(Xte_tok)
    return metrics


def run_sequence_experiment(df_raw, prefix_points=10, seq_prefix_points=30):
    print("\n" + "=" * 60)
    print("轻量序列式实验（固定前缀展平 + Transformer）")
    print("=" * 60)

    X, y = build_sequence_dataset(df_raw, prefix_points=prefix_points)
    print(f"序列样本：{len(X)} 条，输入维度：{X.shape[1] if len(X) else 0}")
    if len(X) < 200:
        print("序列样本过少，跳过该实验。")
        return {}

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    baseline_pred = np.full_like(y_test, y_train.mean(), dtype=float)
    baseline = evaluate_predictions(y_test, baseline_pred)
    print(f"  序列实验 Baseline: MAE={baseline['MAE']:.1f}s")

    seq_models = {
        'SeqMLP': (MLPRegressor(hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=220, random_state=42), True, True),
        'SeqHistGB(log1p)': (HistGradientBoostingRegressor(max_depth=8, learning_rate=0.05, max_iter=220, random_state=42), False, True),
    }

    results = {}
    for name, (model, use_scaled, log_target) in seq_models.items():
        Xtr = X_train_s if use_scaled else X_train
        Xte = X_test_s if use_scaled else X_test
        ytr = np.log1p(y_train) if log_target else y_train

        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        if log_target:
            pred = np.expm1(pred)

        metrics = evaluate_predictions(y_test, pred)
        metrics['improvement'] = (baseline['MAE'] - metrics['MAE']) / baseline['MAE'] * 100
        results[name] = metrics
        print(
            f"  {name:18s}: MAE={metrics['MAE']:.1f}s  RMSE={metrics['RMSE']:.1f}s  "
            f"MAPE={metrics['MAPE']:.1f}%  提升={metrics['improvement']:.1f}%"
        )

    X_tokens, X_meta, y_seq = build_transformer_sequence_dataset(df_raw, prefix_points=seq_prefix_points)
    print(
        f"  Transformer 序列张量：{X_tokens.shape if len(X_tokens) else (0,)}，meta维度：{X_meta.shape[1] if len(X_meta) else 0}，前缀点数：{seq_prefix_points}"
    )
    tf_metrics = fit_seq_transformer(X_tokens, X_meta, y_seq)
    if tf_metrics is not None:
        tf_metrics['improvement'] = (baseline['MAE'] - tf_metrics['MAE']) / baseline['MAE'] * 100
        results['SeqTransformer'] = tf_metrics
        print(
            f"  {'SeqTransformer':18s}: MAE={tf_metrics['MAE']:.1f}s  RMSE={tf_metrics['RMSE']:.1f}s  "
            f"MAPE={tf_metrics['MAPE']:.1f}%  提升={tf_metrics['improvement']:.1f}%"
        )

    if results:
        best_name = min(results, key=lambda k: results[k]['MAE'])
        print(f"  序列实验最优：{best_name}  MAE={results[best_name]['MAE']:.1f}s")

    return results


# ══════════════════════════════════════════════════════════════════════
# 9. 结果摘要写出
# ══════════════════════════════════════════════════════════════════════
def write_summary(clean_stats, tab_results, best_name, prefix_results, seq_results, osm_summary=None):
    path = os.path.join(OUTPUT_DIR, 'taxi_07_experiment_summary.md')

    model_names = [k for k in tab_results.keys() if k != 'Baseline']
    model_lines = [
        f"| {name} | {tab_results[name]['MAE']:.1f} | {tab_results[name]['RMSE']:.1f} | {tab_results[name]['MAPE']:.1f} | {tab_results[name]['improvement']:.1f}% |"
        for name in model_names
    ]

    seq_block = "未运行"
    if seq_results:
        seq_lines = [
            f"| {name} | {seq_results[name]['MAE']:.1f} | {seq_results[name]['RMSE']:.1f} | {seq_results[name]['MAPE']:.1f} | {seq_results[name]['improvement']:.1f}% |"
            for name in seq_results
        ]
        seq_block = "\n".join([
            "| 模型 | MAE (s) | RMSE (s) | MAPE (%) | 相对序列基线提升 |",
            "|---|---:|---:|---:|---:|",
            *seq_lines,
        ])

    prefix_block = "\n".join([f"- 前缀 {pts} 点：MAE = {mae:.1f}s" for pts, mae in sorted(prefix_results.items())]) if prefix_results else "- 未运行"
    osm_block = "- 未启用 OSM 轻量地图语义特征"
    if osm_summary:
        osm_block = "\n".join([
            f"- OSM JSON：{osm_summary.get('osm_json_path', '')}",
            f"- 纳入索引道路节点：{osm_summary.get('node_count', 0)}",
            f"- 纳入索引道路 way：{osm_summary.get('way_count', 0)}",
            f"- 新增地图特征：{', '.join(osm_summary.get('feature_names', []))}",
        ])

    content = f"""# Taxi 实验增强版摘要

## 1. 数据清洗
- 原始读取：{clean_stats.get('raw_rows', 0)} 条
- 缺失过滤后：{clean_stats.get('after_missing_filter', 0)} 条
- 清洗后有效样本：{clean_stats.get('after_clean', 0)} 条
- 累计移除孤立飘点：{clean_stats.get('removed_spikes_total', 0)} 个
- 丢弃 parse_or_bbox：{clean_stats.get('parse_or_bbox', 0)}
- 丢弃 too_many_fast_segments：{clean_stats.get('too_many_fast_segments', 0)}
- 丢弃 extreme_speed：{clean_stats.get('extreme_speed', 0)}
- 丢弃 duration_out_of_range：{clean_stats.get('duration_out_of_range', 0)}

清洗策略：
- 过滤 Porto 合理坐标范围之外的点
- 删除典型“跳远又跳回”的孤立飘点
- 过滤大量超速段（>120 km/h）或极端异常速度轨迹

## 2. 增强版表格模型结果
最优模型：**{best_name}**

| 模型 | MAE (s) | RMSE (s) | MAPE (%) | 相对基线提升 |
|---|---:|---:|---:|---:|
{chr(10).join(model_lines)}

## 3. 前缀长度分析
{prefix_block}

## 4. 轻量序列式实验
{seq_block}

## 5. OSM 轻量地图语义特征
{osm_block}

## 6. 本轮新增特征
- 速度统计：均值、标准差、最大/最小值、最近 3/5 段均速
- 轨迹动态：转向均值、转向次数、加速度波动
- 轨迹形状：曲折度（detour ratio）、静止比例
- 空间网格：起点/前缀末端所在网格

## 7. 产出文件
- taxi_01_prediction.png
- taxi_02_model_comparison.png
- taxi_03_prefix_analysis.png
- taxi_04_error_analysis.png
- taxi_05_feature_importance.png
- taxi_06_sequence_experiment.png
- taxi_07_experiment_summary.md
"""

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"实验摘要已写入：{path}")


def write_osm_comparison_summary(path, baseline_results, baseline_best_name, osm_results, osm_best_name, osm_summary, commands):
    lines = [
        '# Taxi OSM 轻量地图语义特征对比',
        '',
        '## 1. OSM 特征实现概述',
        f"- 数据源：{osm_summary.get('osm_json_path', '')}",
        f"- 轻量索引：规则网格桶（cell={OSMFeatureExtractor.CELL_SIZE_M:.0f}m）+ 道路节点点云近邻近似",
        '- 道路粗类：major / secondary / tertiary / local / nonmotorized',
        f"- 纳入索引道路节点：{osm_summary.get('node_count', 0)}",
        f"- 纳入索引道路 way：{osm_summary.get('way_count', 0)}",
        f"- 新增特征：{', '.join(osm_summary.get('feature_names', []))}",
        '',
        '## 2. 真实运行命令',
        *[f'- `{cmd}`' for cmd in commands],
        '',
        '## 3. 表格模型结果对比',
        '| 实验 | 最优模型 | MAE (s) | RMSE (s) | MAPE (%) |',
        '|---|---|---:|---:|---:|',
        f"| 不加 OSM | {baseline_best_name} | {baseline_results[baseline_best_name]['MAE']:.1f} | {baseline_results[baseline_best_name]['RMSE']:.1f} | {baseline_results[baseline_best_name]['MAPE']:.1f} |",
        f"| 加 OSM | {osm_best_name} | {osm_results[osm_best_name]['MAE']:.1f} | {osm_results[osm_best_name]['RMSE']:.1f} | {osm_results[osm_best_name]['MAPE']:.1f} |",
        '',
    ]
    delta_mae = baseline_results[baseline_best_name]['MAE'] - osm_results[osm_best_name]['MAE']
    rel = delta_mae / baseline_results[baseline_best_name]['MAE'] * 100 if baseline_results[baseline_best_name]['MAE'] else 0.0
    judgment = '有增益' if delta_mae > 0 else '未带来增益'
    lines.extend([
        '## 4. 判断',
        f"- 最优 MAE 变化：{delta_mae:+.1f}s（{rel:+.2f}%）",
        f"- 结论：轻量地图语义特征**{judgment}**。",
        '- 说明：这里使用的是道路节点近邻近似，不做严格 map matching，因此增益更偏向“起终点道路环境”而非精确路径约束。',
        '',
    ])
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"OSM 对比摘要已写入：{path}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['synthetic', 'real'], default='synthetic')
    parser.add_argument('--data', default='train.csv', help='真实数据 CSV 路径')
    parser.add_argument('--n', type=int, default=5000, help='合成数据条数')
    parser.add_argument('--prefix-points', type=int, default=10, help='真实数据默认使用前多少个轨迹点作为已观察前缀')
    parser.add_argument('--seq-prefix-points', type=int, default=30, help='SeqTransformer 使用的前缀点数，可独立于表格特征前缀配置')
    parser.add_argument('--skip-prefix-analysis', action='store_true', help='跳过前缀长度分析（较慢）')
    parser.add_argument('--skip-sequence-experiment', action='store_true', help='跳过轻量序列式实验')
    parser.add_argument('--use-osm-features', action='store_true', help='启用轻量 OSM 地图语义特征')
    parser.add_argument('--osm-json', default='data/porto_highways_bbox.osm.json', help='OSM JSON 路网文件路径')
    args = parser.parse_args()

    print("=" * 60)
    print("出租车行程时长预测 —— Task 4（增强版）")
    print("=" * 60)

    clean_stats = {}
    if args.mode == 'synthetic':
        df_raw = generate_synthetic_data(n=args.n)
    else:
        data_path = args.data
        if not os.path.exists(data_path):
            data_path = os.path.join(os.path.dirname(__file__), args.data)
        df_raw, clean_stats = load_real_data(data_path)

    osm_extractor = None
    osm_summary = None
    if args.use_osm_features:
        osm_path = args.osm_json
        if not os.path.exists(osm_path):
            osm_path = os.path.join(os.path.dirname(__file__), args.osm_json)
        osm_extractor = OSMFeatureExtractor(osm_path)
        osm_summary = {
            'osm_json_path': osm_path,
            'node_count': osm_extractor.node_count,
            'way_count': osm_extractor.way_count,
            'feature_names': [
                'start_nearest_major_m',
                'end_nearest_major_m',
                'start_nearest_secondary_m',
                'end_nearest_secondary_m',
                'start_local_road_count_r200',
                'end_local_road_count_r200',
                'start_major_road_count_r300',
                'end_major_road_count_r300',
                'start_node_density_r200',
                'end_node_density_r200',
                'start_dominant_road_type_enc',
                'end_dominant_road_type_enc',
            ],
        }

    if args.mode == 'synthetic':
        feat_df = extract_features(df_raw, prefix_ratio=0.4, prefix_points=None, osm_extractor=osm_extractor)
    else:
        feat_df = extract_features(df_raw, prefix_points=args.prefix_points, osm_extractor=osm_extractor)

    tab_results, best_name, feat_cols, scaler = train_and_evaluate(feat_df)

    prefix_results = {} if args.skip_prefix_analysis else prefix_analysis(df_raw, osm_extractor=osm_extractor)

    test_err_df = error_analysis(feat_df, tab_results, best_name)

    seq_results = {}
    if not args.skip_sequence_experiment and args.mode == 'real':
        seq_results = run_sequence_experiment(
            df_raw,
            prefix_points=args.prefix_points,
            seq_prefix_points=args.seq_prefix_points,
        )

    visualize_all(feat_df, tab_results, best_name, prefix_results, test_err_df, seq_results=seq_results)

    if args.mode == 'real':
        write_summary(clean_stats, tab_results, best_name, prefix_results, seq_results, osm_summary=osm_summary)

    print("\n" + "=" * 60)
    print("完成！输出目录：", OUTPUT_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()
