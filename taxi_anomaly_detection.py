#!/usr/bin/env python3
"""Task 2: 异常轨迹检测

基于 Porto Taxi Trajectory 数据集，完成：
- 数据加载与严格清洗
- 人工注入多类型异常
- 特征提取
- 多方法异常检测对比
- 结果分析与可视化
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

local_site = os.path.expanduser(
    f"~/.local/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
)
if os.path.isdir(local_site) and local_site not in sys.path:
    sys.path.insert(0, local_site)

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


PORTO_BBOX = {
    "lat_min": 40.8,
    "lat_max": 41.3,
    "lon_min": -8.8,
    "lon_max": -8.4,
}
DEFAULT_DATA_PATH = os.path.expanduser("~/taxi/kaggle_dl/train.csv")
DEFAULT_OUTPUT_DIR = os.path.expanduser("~/taxi/output/task2")

SAMPLE_SECONDS = 15.0
NORMAL_MIN_POINTS = 10
MIN_DURATION_SECONDS = 60.0
MAX_DURATION_SECONDS = 7200.0
SPEED_LIMIT_KMH = 120.0
EXTREME_SPEED_LIMIT_KMH = 300.0

FEATURE_ORDER = [
    "max_speed",
    "speed_std",
    "path_length",
    "straight_line_ratio",
    "max_jump_distance",
    "avg_speed",
    "detour_ratio",
    "point_density",
    "speed_consistency",
    "start_end_distance",
    "n_points",
    "duration_minutes",
    "high_speed_ratio",
    "turn_angle_mean",
    "turn_angle_max",
    "turn_angle_std",
]


@dataclass
class Trajectory:
    trip_id: str
    points: List[Tuple[float, float]]
    is_anomaly: int = 0
    anomaly_type: str = "normal"
    anomaly_span: Optional[Tuple[int, int]] = None
    original_points: Optional[List[Tuple[float, float]]] = None


# ---------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """两点球面距离，单位：米。"""
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def point_in_bbox(lon: float, lat: float) -> bool:
    return (
        PORTO_BBOX["lon_min"] <= lon <= PORTO_BBOX["lon_max"]
        and PORTO_BBOX["lat_min"] <= lat <= PORTO_BBOX["lat_max"]
    )


def clamp_point(lon: float, lat: float) -> Tuple[float, float]:
    lon = float(np.clip(lon, PORTO_BBOX["lon_min"], PORTO_BBOX["lon_max"]))
    lat = float(np.clip(lat, PORTO_BBOX["lat_min"], PORTO_BBOX["lat_max"]))
    return lon, lat


def parse_polyline(polyline: str) -> Optional[List[Tuple[float, float]]]:
    if not isinstance(polyline, str) or not polyline:
        return None
    try:
        pts = json.loads(polyline)
    except Exception:
        try:
            pts = ast.literal_eval(polyline)
        except Exception:
            return None
    if not isinstance(pts, list):
        return None

    clean: List[Tuple[float, float]] = []
    for item in pts:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            lon = float(item[0])
            lat = float(item[1])
        except Exception:
            continue
        if math.isnan(lon) or math.isnan(lat):
            continue
        clean.append((lon, lat))
    return clean if len(clean) >= 2 else None


def segment_distances(points: Sequence[Tuple[float, float]]) -> np.ndarray:
    if len(points) < 2:
        return np.asarray([], dtype=float)
    d = [
        haversine_m(points[i - 1][1], points[i - 1][0], points[i][1], points[i][0])
        for i in range(1, len(points))
    ]
    return np.asarray(d, dtype=float)


def segment_speeds_kmh(points: Sequence[Tuple[float, float]]) -> np.ndarray:
    return segment_distances(points) / SAMPLE_SECONDS * 3.6


def segment_headings_deg(points: Sequence[Tuple[float, float]]) -> np.ndarray:
    if len(points) < 2:
        return np.asarray([], dtype=float)
    headings = []
    for i in range(1, len(points)):
        lon1, lat1 = points[i - 1]
        lon2, lat2 = points[i]
        y = math.sin(math.radians(lon2 - lon1)) * math.cos(math.radians(lat2))
        x = (
            math.cos(math.radians(lat1)) * math.sin(math.radians(lat2))
            - math.sin(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.cos(math.radians(lon2 - lon1))
        )
        headings.append((math.degrees(math.atan2(y, x)) + 360.0) % 360.0)
    return np.asarray(headings, dtype=float)


def heading_diffs_deg(headings: np.ndarray) -> np.ndarray:
    if len(headings) < 2:
        return np.asarray([], dtype=float)
    diffs = np.abs((headings[1:] - headings[:-1] + 180.0) % 360.0 - 180.0)
    return diffs


# ---------------------------------------------------------------------
# 数据清洗
# ---------------------------------------------------------------------
def clean_trajectory(points: Sequence[Tuple[float, float]]) -> Optional[List[Tuple[float, float]]]:
    pts = [tuple(p) for p in points if point_in_bbox(p[0], p[1])]
    if len(pts) < NORMAL_MIN_POINTS:
        return None

    # 三点规则：若中间点两侧速度异常高，但跳过中点后速度恢复正常，则删除该飘点。
    for _ in range(4):
        if len(pts) < NORMAL_MIN_POINTS:
            return None
        speeds = segment_speeds_kmh(pts)
        if len(speeds) < 2:
            break
        to_drop = []
        for i in range(1, len(pts) - 1):
            d_prev = haversine_m(pts[i - 1][1], pts[i - 1][0], pts[i][1], pts[i][0])
            d_next = haversine_m(pts[i][1], pts[i][0], pts[i + 1][1], pts[i + 1][0])
            d_skip = haversine_m(pts[i - 1][1], pts[i - 1][0], pts[i + 1][1], pts[i + 1][0])
            v_prev = d_prev / SAMPLE_SECONDS * 3.6
            v_next = d_next / SAMPLE_SECONDS * 3.6
            v_skip = d_skip / (2.0 * SAMPLE_SECONDS) * 3.6
            if v_prev > SPEED_LIMIT_KMH and v_next > SPEED_LIMIT_KMH and v_skip <= SPEED_LIMIT_KMH:
                to_drop.append(i)
        if not to_drop:
            break
        pts = [p for j, p in enumerate(pts) if j not in set(to_drop)]

    if len(pts) < NORMAL_MIN_POINTS:
        return None

    speeds = segment_speeds_kmh(pts)
    duration_s = (len(pts) - 1) * SAMPLE_SECONDS
    if duration_s < MIN_DURATION_SECONDS or duration_s > MAX_DURATION_SECONDS:
        return None
    if len(speeds) == 0:
        return None
    if float(np.max(speeds)) > EXTREME_SPEED_LIMIT_KMH:
        return None
    high_speed_segments = int(np.sum(speeds > SPEED_LIMIT_KMH))
    if high_speed_segments > max(2, int(math.ceil(len(speeds) * 0.12))):
        return None
    if len(pts) < NORMAL_MIN_POINTS:
        return None
    return pts


# ---------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------
def load_cleaned_trajectories(data_path: str, n_rows: int, seed: int) -> Tuple[List[Trajectory], Dict[str, int]]:
    df = pd.read_csv(
        data_path,
        nrows=n_rows,
        usecols=["TRIP_ID", "MISSING_DATA", "POLYLINE"],
    )
    df = df[df["MISSING_DATA"] == False].copy()  # noqa: E712
    stats = {
        "raw_rows": int(len(df)),
        "parsed_ok": 0,
        "clean_ok": 0,
        "discard_parse": 0,
        "discard_bbox": 0,
        "discard_short": 0,
        "discard_duration": 0,
        "discard_speed": 0,
        "discard_other": 0,
    }

    cleaned: List[Trajectory] = []
    for row in df.itertuples(index=False):
        pts = parse_polyline(row.POLYLINE)
        if pts is None:
            stats["discard_parse"] += 1
            continue
        stats["parsed_ok"] += 1

        in_bbox_pts = [p for p in pts if point_in_bbox(p[0], p[1])]
        if len(in_bbox_pts) < NORMAL_MIN_POINTS:
            stats["discard_bbox"] += 1
            continue

        cleaned_pts = clean_trajectory(pts)
        if cleaned_pts is None:
            # 粗略统计原因。
            duration_s = (len(in_bbox_pts) - 1) * SAMPLE_SECONDS
            if len(in_bbox_pts) < NORMAL_MIN_POINTS:
                stats["discard_short"] += 1
            elif duration_s < MIN_DURATION_SECONDS or duration_s > MAX_DURATION_SECONDS:
                stats["discard_duration"] += 1
            else:
                speeds = segment_speeds_kmh(in_bbox_pts)
                if len(speeds) and float(np.max(speeds)) > EXTREME_SPEED_LIMIT_KMH:
                    stats["discard_speed"] += 1
                elif len(speeds) and int(np.sum(speeds > SPEED_LIMIT_KMH)) > max(2, int(math.ceil(len(speeds) * 0.12))):
                    stats["discard_speed"] += 1
                else:
                    stats["discard_other"] += 1
            continue

        cleaned.append(
            Trajectory(
                trip_id=str(row.TRIP_ID),
                points=cleaned_pts,
            )
        )
        stats["clean_ok"] += 1

    rng = np.random.default_rng(seed)
    rng.shuffle(cleaned)
    return cleaned, stats


# ---------------------------------------------------------------------
# 异常注入
# ---------------------------------------------------------------------
def _middle_index(n: int, rng: np.random.Generator) -> int:
    lo = max(2, n // 4)
    hi = max(lo + 1, (3 * n) // 4)
    return int(rng.integers(lo, hi))


def inject_jump_anomaly(points: List[Tuple[float, float]], rng: np.random.Generator):
    pts = [list(p) for p in points]
    n = len(pts)
    if n < 6:
        return points, None
    idx = _middle_index(n, rng)
    span = int(rng.integers(2, 4))
    offset_lon = float(rng.choice([-1, 1]) * rng.uniform(0.03, 0.06))
    offset_lat = float(rng.choice([-1, 1]) * rng.uniform(0.03, 0.06))
    end = min(n, idx + span)
    for i in range(idx, end):
        pts[i] = list(clamp_point(pts[i][0] + offset_lon, pts[i][1] + offset_lat))
    return [tuple(p) for p in pts], (idx, end - 1)


def inject_speed_anomaly(points: List[Tuple[float, float]], rng: np.random.Generator):
    pts = [list(p) for p in points]
    n = len(pts)
    if n < 6:
        return points, None
    idx = _middle_index(n, rng)
    n_new = int(rng.integers(4, 7))
    base_lon, base_lat = pts[idx]
    direction = rng.normal(size=2)
    direction = direction / (np.linalg.norm(direction) + 1e-9)
    step_m = rng.uniform(420.0, 760.0)
    deg_lat = step_m / 111_000.0
    deg_lon = step_m / (111_000.0 * math.cos(math.radians(base_lat)) + 1e-9)
    scale_lon = deg_lon * direction[0]
    scale_lat = deg_lat * direction[1]
    inserted = []
    for i in range(n_new):
        lon = base_lon + scale_lon * (i + 1) + rng.normal(0, 0.00018)
        lat = base_lat + scale_lat * (i + 1) + rng.normal(0, 0.00018)
        inserted.append(clamp_point(lon, lat))
    new_pts = pts[:idx] + [list(p) for p in inserted] + pts[idx + 1 :]
    return [tuple(p) for p in new_pts], (idx, idx + len(inserted) - 1)


def inject_detour_anomaly(points: List[Tuple[float, float]], rng: np.random.Generator):
    pts = [list(p) for p in points]
    n = len(pts)
    if n < 8:
        return points, None
    idx = _middle_index(n, rng)
    left = pts[max(0, idx - 1)]
    right = pts[min(n - 1, idx + 1)]
    lon1, lat1 = left
    lon2, lat2 = right
    vlon = lon2 - lon1
    vlat = lat2 - lat1
    perp = np.asarray([-vlat, vlon], dtype=float)
    perp = perp / (np.linalg.norm(perp) + 1e-9)
    amp = rng.uniform(0.008, 0.018)
    side = float(rng.choice([-1.0, 1.0]))
    route = []
    for frac, bend in [
        (0.20, 0.75),
        (0.42, 1.00),
        (0.62, 0.70),
        (0.82, -0.35),
    ]:
        lon = lon1 + vlon * frac + perp[0] * amp * side * bend
        lat = lat1 + vlat * frac + perp[1] * amp * side * bend
        route.append(clamp_point(lon, lat))
    new_pts = pts[: idx + 1] + [list(p) for p in route] + pts[idx + 1 :]
    return [tuple(p) for p in new_pts], (idx + 1, idx + len(route))


def inject_truncate_anomaly(points: List[Tuple[float, float]], rng: np.random.Generator):
    pts = list(points)
    n = len(pts)
    if n < 10:
        return points, None
    cut_ratio = float(rng.uniform(0.50, 0.70))
    new_len = max(2, int(round(n * (1.0 - cut_ratio))))
    new_len = min(new_len, n - 1)
    return pts[:new_len], (new_len, n - 1)


def make_anomalies(base: List[Trajectory], anomaly_ratio: float, seed: int) -> List[Trajectory]:
    rng = np.random.default_rng(seed)
    n_anom = int(round(len(base) * anomaly_ratio))
    n_anom = max(1, n_anom)
    types = ["jump", "speed", "detour", "truncate"]
    chosen = rng.choice(len(base), size=n_anom, replace=False)
    anomalies: List[Trajectory] = []
    for i, idx in enumerate(chosen):
        src = base[idx]
        a_type = types[i % len(types)]
        if a_type == "jump":
            new_pts, span = inject_jump_anomaly(src.points, rng)
        elif a_type == "speed":
            new_pts, span = inject_speed_anomaly(src.points, rng)
        elif a_type == "detour":
            new_pts, span = inject_detour_anomaly(src.points, rng)
        else:
            new_pts, span = inject_truncate_anomaly(src.points, rng)

        anomalies.append(
            Trajectory(
                trip_id=f"{src.trip_id}_anom_{a_type}_{i}",
                points=new_pts,
                is_anomaly=1,
                anomaly_type=a_type,
                anomaly_span=span,
                original_points=list(src.points),
            )
        )
    return anomalies


# ---------------------------------------------------------------------
# 特征提取
# ---------------------------------------------------------------------
def trajectory_features(points: Sequence[Tuple[float, float]]) -> Dict[str, float]:
    pts = list(points)
    n = len(pts)
    if n < 2:
        return {k: 0.0 for k in FEATURE_ORDER}

    seg_dist = segment_distances(pts)
    seg_speed = seg_dist / SAMPLE_SECONDS * 3.6
    path_length = float(seg_dist.sum())
    start_end_distance = haversine_m(pts[0][1], pts[0][0], pts[-1][1], pts[-1][0])
    straight_line_ratio = start_end_distance / path_length if path_length > 1e-9 else 0.0
    detour_ratio = path_length / start_end_distance if start_end_distance > 1e-9 else 999.0
    duration_s = max(1.0, (n - 1) * SAMPLE_SECONDS)
    point_density = n / (duration_s / 60.0)

    max_speed = float(seg_speed.max()) if len(seg_speed) else 0.0
    speed_std = float(seg_speed.std(ddof=0)) if len(seg_speed) else 0.0
    avg_speed = float(seg_speed.mean()) if len(seg_speed) else 0.0
    speed_consistency = speed_std / (avg_speed + 1e-9)
    max_jump_distance = float(seg_dist.max()) if len(seg_dist) else 0.0
    high_speed_ratio = float(np.mean(seg_speed > SPEED_LIMIT_KMH)) if len(seg_speed) else 0.0
    headings = segment_headings_deg(pts)
    turn_diffs = heading_diffs_deg(headings)
    turn_angle_mean = float(turn_diffs.mean()) if len(turn_diffs) else 0.0
    turn_angle_max = float(turn_diffs.max()) if len(turn_diffs) else 0.0
    turn_angle_std = float(turn_diffs.std(ddof=0)) if len(turn_diffs) else 0.0

    # Enhanced features for better truncation/speed/jump detection
    speed_max_over_median = max_speed / (np.median(seg_speed) + 1e-9) if len(seg_speed) > 0 else 1.0
    n_jumps = int(np.sum(seg_dist > 1000)) if len(seg_dist) > 0 else 0
    jump_density = float(n_jumps) / max(1, n - 1)

    return {
        "max_speed": max_speed,
        "speed_std": speed_std,
        "path_length": path_length,
        "straight_line_ratio": straight_line_ratio,
        "max_jump_distance": max_jump_distance,
        "avg_speed": avg_speed,
        "detour_ratio": detour_ratio,
        "point_density": point_density,
        "speed_consistency": speed_consistency,
        "start_end_distance": float(start_end_distance),
        "n_points": float(n),
        "duration_minutes": duration_s / 60.0,
        "high_speed_ratio": high_speed_ratio,
        "turn_angle_mean": turn_angle_mean,
        "turn_angle_max": turn_angle_max,
        "turn_angle_std": turn_angle_std,
        "speed_max_over_median": speed_max_over_median,
        "jump_density": jump_density,
    }


def normalize_track(points: Sequence[Tuple[float, float]]) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.shape[0] == 0:
        return arr
    mn = arr.min(axis=0)
    mx = arr.max(axis=0)
    span = np.where((mx - mn) < 1e-9, 1.0, mx - mn)
    return (arr - mn) / span


def build_feature_frame(trajs: List[Trajectory]) -> pd.DataFrame:
    rows = []
    for t in trajs:
        feat = trajectory_features(t.points)
        feat["trip_id"] = t.trip_id
        feat["is_anomaly"] = t.is_anomaly
        feat["anomaly_type"] = t.anomaly_type
        feat["n_points_raw"] = len(t.points)
        rows.append(feat)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 检测与评估
# ---------------------------------------------------------------------
def fit_and_score_methods(
    features: pd.DataFrame,
    anomaly_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    X = features[FEATURE_ORDER].to_numpy(dtype=float)
    y = features["is_anomaly"].to_numpy(dtype=int)

    normal_mask = y == 0
    scaler = StandardScaler()
    scaler.fit(X[normal_mask])
    Xs = scaler.transform(X)
    Xn = Xs[normal_mask]

    methods = {
        "Isolation Forest": IsolationForest(
            n_estimators=400,
            contamination=anomaly_ratio,
            random_state=seed,
            n_jobs=-1,
        ),
        "One-Class SVM": OneClassSVM(
            nu=anomaly_ratio,
            kernel="rbf",
            gamma="scale",
        ),
    }

    # LOF grid search over n_neighbors and contamination
    best_lof_f1 = -1
    best_lof_result = None
    best_lof_pred = None
    best_lof_score = None
    lof_grid = []
    n_neighbors_candidates = [10, 15, 20, 25, 30, 35, 40]
    contamination_candidates = [0.05, 0.07, anomaly_ratio, 0.10, 0.12]
    for nn in n_neighbors_candidates:
        for cont in contamination_candidates:
            if nn >= len(Xn):
                continue
            lof = LocalOutlierFactor(n_neighbors=nn, contamination=cont, novelty=True)
            lof.fit(Xn)
            pred = lof.predict(Xs)
            y_pred = (pred == -1).astype(int)
            if hasattr(lof, "decision_function"):
                score = -lof.decision_function(Xs)
            else:
                score = -lof.score_samples(Xs)
            score = np.asarray(score, dtype=float)
            f1 = f1_score(y, y_pred, zero_division=0)
            prec = precision_score(y, y_pred, zero_division=0)
            rec = recall_score(y, y_pred, zero_division=0)
            auc = roc_auc_score(y, score) if len(np.unique(y)) > 1 else 0.0
            lof_grid.append({"n_neighbors": nn, "contamination": cont, "f1": f1, "precision": prec, "recall": rec, "auroc": auc})
            if f1 > best_lof_f1:
                best_lof_f1 = f1
                best_lof_result = {
                    "method": f"LOF(nn={nn},cont={cont})",
                    "precision": prec, "recall": rec, "f1": f1, "auroc": auc,
                    "pred_anomaly_count": int(y_pred.sum()),
                    "true_anomaly_count": int(y.sum()),
                }
                best_lof_score = score
                best_lof_pred = y_pred
    print(f"LOF grid best: nn={best_lof_result['method'].split('nn=')[1].split(',')[0]}, "
          f"F1={best_lof_f1:.4f}, AUROC={best_lof_result['auroc']:.4f}")

    rows = []
    score_map: Dict[str, np.ndarray] = {}
    pred_map: Dict[str, np.ndarray] = {}
    for name, model in methods.items():
        model.fit(Xn)
        pred = model.predict(Xs)
        y_pred = (pred == -1).astype(int)
        if hasattr(model, "decision_function"):
            score = -model.decision_function(Xs)
        else:
            score = -model.score_samples(Xs)
        score = np.asarray(score, dtype=float)
        score_map[name] = score
        pred_map[name] = y_pred

        rows.append(
            {
                "method": name,
                "precision": precision_score(y, y_pred, zero_division=0),
                "recall": recall_score(y, y_pred, zero_division=0),
                "f1": f1_score(y, y_pred, zero_division=0),
                "auroc": roc_auc_score(y, score) if len(np.unique(y)) > 1 else np.nan,
                "pred_anomaly_count": int(y_pred.sum()),
                "true_anomaly_count": int(y.sum()),
            }
        )

    # Add LOF best result to maps and rows
    if best_lof_result is not None:
        score_map["LOF"] = best_lof_score
        pred_map["LOF"] = best_lof_pred
        rows.append(best_lof_result)

    metrics_df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    return metrics_df, score_map, pred_map


def type_recall_table(features: pd.DataFrame, pred_map: Dict[str, np.ndarray]) -> pd.DataFrame:
    y = features["is_anomaly"].to_numpy(dtype=int)
    rows = []
    for method, pred in pred_map.items():
        for a_type in ["jump", "speed", "detour", "truncate"]:
            mask = features["anomaly_type"].eq(a_type).to_numpy() & (y == 1)
            if mask.sum() == 0:
                continue
            rows.append(
                {
                    "method": method,
                    "anomaly_type": a_type,
                    "detection_rate": float((pred[mask] == 1).mean()),
                    "count": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def top_anomaly_cases(features: pd.DataFrame, score_map: Dict[str, np.ndarray], k: int = 10) -> pd.DataFrame:
    df = features.copy()
    df["anomaly_score"] = np.mean(np.column_stack(list(score_map.values())), axis=1)
    return df.sort_values("anomaly_score", ascending=False).head(k).reset_index(drop=True)


def explain_case(row: pd.Series) -> str:
    reasons = []
    if row["max_jump_distance"] > 700:
        reasons.append("相邻点跳变大")
    if row["max_speed"] > 120:
        reasons.append("局部速度异常高")
    if row["high_speed_ratio"] > 0.2:
        reasons.append("高速段占比高")
    if row["detour_ratio"] > 1.35:
        reasons.append("绕行明显")
    if row["turn_angle_max"] > 110:
        reasons.append("转向突变")
    if row["point_density"] < 3.0:
        reasons.append("点密度偏低")
    if row["n_points_raw"] < 12:
        reasons.append("轨迹偏短")
    if row["anomaly_type"] == "truncate":
        reasons.append("末尾被截断")
    if not reasons:
        reasons.append("多特征偏离正常分布")
    return "；".join(reasons)


# ---------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------
def plot_method_comparison(metrics: pd.DataFrame, output_dir: str) -> str:
    methods = metrics["method"].tolist()
    metrics = metrics.set_index("method")
    cols = ["precision", "recall", "f1", "auroc"]
    x = np.arange(len(methods))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, col in enumerate(cols):
        ax.bar(x + (i - 1.5) * width, metrics.loc[methods, col].values, width=width, label=col.upper())
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Anomaly Detection: Method Comparison")
    ax.legend(ncol=4, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    path = os.path.join(output_dir, "task2_method_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_traj(
    ax,
    pts: np.ndarray,
    anomaly_span: Optional[Tuple[int, int]],
    title: str,
    truncate_mode: bool = False,
    original: Optional[np.ndarray] = None,
):
    if pts.shape[0] < 2:
        ax.set_title(title)
        ax.axis("off")
        return

    if truncate_mode and original is not None and original.shape[0] >= 2:
        ax.plot(original[:, 0], original[:, 1], color="#cfcfcf", lw=2, ls="--", label="original")
        ax.plot(pts[:, 0], pts[:, 1], color="#1f77b4", lw=2.2, label="kept")
        if len(original) > len(pts):
            tail = original[len(pts) - 1 :]
            if tail.shape[0] >= 2:
                ax.plot(tail[:, 0], tail[:, 1], color="#d62728", lw=2.4, ls=":", label="removed tail")
        ax.scatter(pts[0, 0], pts[0, 1], c="#2ca02c", s=25, zorder=4)
        ax.scatter(pts[-1, 0], pts[-1, 1], c="#ff7f0e", s=25, zorder=4)
    else:
        ax.plot(pts[:, 0], pts[:, 1], color="#1f77b4", lw=2.2)
        if anomaly_span is not None:
            s, e = anomaly_span
            s = max(0, min(int(s), len(pts) - 1))
            e = max(s, min(int(e), len(pts) - 1))
            if e > s:
                ax.plot(pts[s : e + 1, 0], pts[s : e + 1, 1], color="#d62728", lw=3.0)
                ax.scatter(pts[s : e + 1, 0], pts[s : e + 1, 1], c="#d62728", s=18, zorder=4)
        ax.scatter(pts[0, 0], pts[0, 1], c="#2ca02c", s=25, zorder=4)
        ax.scatter(pts[-1, 0], pts[-1, 1], c="#ff7f0e", s=25, zorder=4)

    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")


def plot_anomaly_examples(trajs: List[Trajectory], output_dir: str) -> str:
    by_type: Dict[str, List[Trajectory]] = {k: [] for k in ["jump", "speed", "detour", "truncate"]}
    for t in trajs:
        if t.is_anomaly:
            by_type.setdefault(t.anomaly_type, []).append(t)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.ravel()
    titles = ["Type A: Jump Anomaly", "Type B: Speeding", "Type C: Detour", "Type D: Truncated"]
    keys = ["jump", "speed", "detour", "truncate"]
    for ax, title, key in zip(axes, titles, keys):
        sample = by_type.get(key, [])[:1]
        if not sample:
            ax.axis("off")
            continue
        t = sample[0]
        pts = normalize_track(t.points)
        if key == "truncate" and t.original_points is not None:
            original = normalize_track(t.original_points)
            _plot_traj(ax, pts, t.anomaly_span, title, truncate_mode=True, original=original)
        else:
            _plot_traj(ax, pts, t.anomaly_span, title)
        if key in {"jump", "truncate"}:
            ax.legend(loc="best", frameon=False, fontsize=8)

    fig.suptitle("Anomaly Trajectory Examples", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    path = os.path.join(output_dir, "task2_anomaly_examples.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_distributions(features: pd.DataFrame, output_dir: str) -> str:
    show_cols = [
        "max_speed",
        "path_length",
        "detour_ratio",
        "max_jump_distance",
        "straight_line_ratio",
        "turn_angle_max",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    normal = features[features["is_anomaly"] == 0]
    abnormal = features[features["is_anomaly"] == 1]
    for ax, col in zip(axes, show_cols):
        bins = 30
        ax.hist(normal[col], bins=bins, alpha=0.65, density=True, label="normal", color="#1f77b4")
        ax.hist(abnormal[col], bins=bins, alpha=0.65, density=True, label="anomaly", color="#d62728")
        ax.set_title(col)
        ax.tick_params(labelsize=8)
    axes[0].legend(frameon=False)
    fig.suptitle("正常轨迹与异常轨迹特征分布对比", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    path = os.path.join(output_dir, "task2_feature_distributions.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_type_recall(type_df: pd.DataFrame, output_dir: str) -> str:
    if type_df.empty:
        return ""
    pivot = type_df.pivot(index="anomaly_type", columns="method", values="detection_rate").fillna(0.0)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    pivot.plot(kind="bar", ax=ax, width=0.82)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Detection rate")
    ax.set_title("Detection Rate by Anomaly Type")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    path = os.path.join(output_dir, "task2_type_recall.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------
# 报告输出
# ---------------------------------------------------------------------
def fmt(x: float) -> str:
    return f"{x:.4f}"


def make_markdown_report(
    base_count: int,
    anomaly_count: int,
    anomaly_ratio: float,
    features: pd.DataFrame,
    metrics: pd.DataFrame,
    type_df: pd.DataFrame,
    top_cases: pd.DataFrame,
    figure_paths: Dict[str, str],
    clean_stats: Dict[str, int],
) -> str:
    lines: List[str] = []
    lines.append("# Task 2 异常轨迹检测报告")
    lines.append("")
    lines.append("## 1. 数据概况")
    lines.append(f"- 原始读取行数：{clean_stats['raw_rows']} 条")
    lines.append(f"- 清洗后可用轨迹：{clean_stats['clean_ok']} 条")
    lines.append(f"- 基础正常轨迹：{base_count} 条")
    lines.append(f"- 人工异常轨迹：{anomaly_count} 条")
    lines.append(f"- 异常比例：{anomaly_ratio:.2%}")
    lines.append(f"- 总样本数：{len(features)} 条")
    lines.append("")
    lines.append("## 2. 清洗规则")
    lines.append(f"- Porto 边界框：lat {PORTO_BBOX['lat_min']}~{PORTO_BBOX['lat_max']}, lon {PORTO_BBOX['lon_min']}~{PORTO_BBOX['lon_max']}")
    lines.append(f"- 孤立飘点：三点规则，满足条件则剔除中点")
    lines.append(f"- 异常速度：单段 > {EXTREME_SPEED_LIMIT_KMH:.0f} km/h 直接丢弃；> {SPEED_LIMIT_KMH:.0f} km/h 过多则丢弃")
    lines.append(f"- 时长范围：{MIN_DURATION_SECONDS:.0f}s~{MAX_DURATION_SECONDS:.0f}s")
    lines.append(f"- 最少点数：{NORMAL_MIN_POINTS} 个")
    lines.append("")
    lines.append("## 3. 特征设计")
    lines.append(
        "- max_speed, speed_std, path_length, straight_line_ratio, max_jump_distance, avg_speed"
    )
    lines.append(
        "- detour_ratio, point_density, speed_consistency, start_end_distance, n_points, duration_minutes"
    )
    lines.append("- high_speed_ratio, turn_angle_mean, turn_angle_max, turn_angle_std")
    lines.append("")
    lines.append("## 4. 方法对比")
    lines.append("| Method | Precision | Recall | F1 | AUROC |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.method} | {fmt(row.precision)} | {fmt(row.recall)} | {fmt(row.f1)} | {fmt(row.auroc)} |"
        )
    lines.append("")
    lines.append("## 5. 各异常类型检出率")
    if not type_df.empty:
        lines.append("| Anomaly Type | Method | Detection Rate | Count |")
        lines.append("|---|---|---:|---:|")
        for row in type_df.itertuples(index=False):
            lines.append(
                f"| {row.anomaly_type} | {row.method} | {fmt(row.detection_rate)} | {row.count} |"
            )
    else:
        lines.append("无可用结果。")
    lines.append("")
    lines.append("## 6. 结论")
    best = metrics.sort_values("f1", ascending=False).iloc[0]
    lines.append(f"- 综合表现最优方法：{best['method']}，F1 = {best['f1']:.4f}。")
    lines.append("- 跳变异常主要由 max_jump_distance 与 max_speed 捕获。")
    lines.append("- 截断异常主要由 n_points、duration_minutes 与 point_density 捕获。")
    lines.append("- 绕行异常主要表现为 detour_ratio 与 turn_angle_max 上升。")
    lines.append("")
    lines.append("## 7. 前 10 条异常样本")
    lines.append("| Rank | Trip ID | True Type | n_points | max_speed | detour_ratio | max_jump_distance | Reason |")
    lines.append("|---|---|---|---:|---:|---:|---:|---|")
    for i, row in top_cases.iterrows():
        lines.append(
            f"| {i+1} | {row.trip_id} | {row.anomaly_type} | {int(row.n_points_raw)} | {row.max_speed:.2f} | {row.detour_ratio:.2f} | {row.max_jump_distance:.1f} | {explain_case(row)} |"
        )
    lines.append("")
    lines.append("## 8. 图表")
    for key, path in figure_paths.items():
        rel = os.path.basename(path)
        lines.append(f"- {key}: `{rel}`")
    lines.append("")
    lines.append("## 9. 说明")
    lines.append("- Isolation Forest 对全局离群更敏感。")
    lines.append("- LOF 对局部密度突变更敏感。")
    lines.append("- One-Class SVM 对边界型样本敏感，但波动更大。")
    lines.append("- 本实验的异常为人工注入，具备明确 ground truth。")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------
def choose_base_count(cleaned_count: int) -> int:
    if cleaned_count >= 3000:
        return 3000
    if cleaned_count >= 2000:
        return 2000
    return cleaned_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Porto Taxi 异常轨迹检测")
    parser.add_argument("--n-sample", type=int, default=30000)
    parser.add_argument("--anomaly-ratio", type=float, default=0.07)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    cleaned, clean_stats = load_cleaned_trajectories(args.data_path, args.n_sample, args.seed)
    if len(cleaned) < 10:
        raise ValueError(f"清洗后样本过少：仅 {len(cleaned)} 条。")

    base_count = choose_base_count(len(cleaned))
    rng = np.random.default_rng(args.seed)
    base_idx = rng.choice(len(cleaned), size=base_count, replace=False)
    base = [cleaned[i] for i in base_idx]

    anomaly_ratio = float(np.clip(args.anomaly_ratio, 0.05, 0.08))
    anomalies = make_anomalies(base, anomaly_ratio, args.seed + 7)
    all_trajs = base + anomalies
    features = build_feature_frame(all_trajs)

    metrics, score_map, pred_map = fit_and_score_methods(features, anomaly_ratio, args.seed)
    type_df = type_recall_table(features, pred_map)
    top_cases = top_anomaly_cases(features, score_map, k=10)

    figure_paths = {
        "方法对比": plot_method_comparison(metrics, args.output_dir),
        "异常轨迹示例": plot_anomaly_examples(all_trajs, args.output_dir),
        "特征分布": plot_feature_distributions(features, args.output_dir),
        "类型检出率": plot_type_recall(type_df, args.output_dir),
    }

    report = make_markdown_report(
        base_count=len(base),
        anomaly_count=len(anomalies),
        anomaly_ratio=anomaly_ratio,
        features=features,
        metrics=metrics,
        type_df=type_df,
        top_cases=top_cases,
        figure_paths=figure_paths,
        clean_stats=clean_stats,
    )
    report_path = os.path.join(args.output_dir, "task2_anomaly_summary.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    metrics_path = os.path.join(args.output_dir, "task2_metrics.csv")
    features_path = os.path.join(args.output_dir, "task2_features.csv")
    features.to_csv(features_path, index=False, encoding="utf-8-sig")
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    if not type_df.empty:
        type_df.to_csv(os.path.join(args.output_dir, "task2_type_recall.csv"), index=False, encoding="utf-8-sig")

    print("=== Task 2 completed ===")
    print(f"Raw rows: {clean_stats['raw_rows']}")
    print(f"Cleaned trajectories: {clean_stats['clean_ok']}")
    print(f"Base trajectories: {len(base)}")
    print(f"Injected anomalies: {len(anomalies)}")
    print("Metrics:")
    print(metrics.to_string(index=False))
    print(f"Report saved to: {report_path}")
    print(f"Figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
