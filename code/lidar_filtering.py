from pathlib import Path
import sys

import numpy as np
import torch


current_dir = Path(__file__).resolve().parent
radelft_dir = current_dir / "radelft"
sys.path.insert(0, str(radelft_dir))
sys.path.insert(0, str(current_dir))

from radelft.data_preparation import data_preparation


def _to_range_doppler_azimuth(power_cube, params):
    expected = (len(params["range_axis"]), 128, len(params["azimuth_axis"]))
    padded = (len(params["range_axis"]) + 12, 128, len(params["azimuth_axis"]) + 16)

    if power_cube.shape == expected or power_cube.shape == padded:
        return power_cube
    if power_cube.shape == (128, expected[0], expected[2]):
        return np.transpose(power_cube, (1, 0, 2))
    if power_cube.shape == (128, padded[0], padded[2]):
        return np.transpose(power_cube, (1, 0, 2))

    raise ValueError(f"Unexpected radar cube shape for LiDAR GT filtering: {power_cube.shape}")


def _crop_power_to_bev_shape(power_cube, params):
    power_cube = _to_range_doppler_azimuth(power_cube, params)
    expected = (len(params["range_axis"]), power_cube.shape[1], len(params["azimuth_axis"]))
    padded = (len(params["range_axis"]) + 12, power_cube.shape[1], len(params["azimuth_axis"]) + 16)

    if power_cube.shape == expected:
        return power_cube
    if power_cube.shape == padded:
        return power_cube[:-12, :, 8:-8]

    raise ValueError(f"Unexpected radar cube shape for LiDAR GT filtering: {power_cube.shape}")


def compute_normalized_spar(power_cube, params):
    cropped = _crop_power_to_bev_shape(power_cube, params)
    doppler_bins = cropped.shape[1]
    max_power = np.max(cropped, axis=1)
    sum_power = np.sum(cropped, axis=1)
    spar = (doppler_bins * max_power / (sum_power + 1e-8) - 1.0) / (doppler_bins - 1.0)
    return np.nan_to_num(spar, nan=0.0, posinf=0.0, neginf=0.0)


def _nearest_axis_indices(values, axis):
    axis = np.asarray(axis)
    indices = np.searchsorted(axis, values, side="left")
    valid = (indices >= 0) & (indices < len(axis))

    valid_positions = np.flatnonzero(valid)
    valid_indices = indices[valid]
    valid_values = values[valid]
    move_left = (valid_indices > 0) & (
        np.abs(valid_values - axis[valid_indices - 1])
        < np.abs(valid_values - axis[valid_indices])
    )
    indices[valid_positions[move_left]] -= 1
    return indices, valid


def _point_to_bev_indices(point_cloud, params):
    spherical = data_preparation.cartesian_to_spherical(
        point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
    )
    range_indices, range_valid = _nearest_axis_indices(spherical[:, 0], params["range_axis"])
    azimuth_indices, azimuth_valid = _nearest_axis_indices(spherical[:, 1], params["azimuth_axis"])
    _, elevation_valid = _nearest_axis_indices(spherical[:, 2], params["elevation_axis"])

    valid = range_valid & azimuth_valid & elevation_valid
    bev_azimuth_indices = len(params["azimuth_axis"]) - 1 - azimuth_indices
    return range_indices, bev_azimuth_indices, valid


def _filter_lidar_points(point_cloud, spar_map, params):
    range_indices, azimuth_indices, valid = _point_to_bev_indices(point_cloud, params)
    high = point_cloud[:, 2] > 2.0
    candidates = high & valid

    keep = np.ones(point_cloud.shape[0], dtype=bool)
    candidate_positions = np.flatnonzero(candidates)
    candidate_values = spar_map[range_indices[candidate_positions], azimuth_indices[candidate_positions]]
    drop_positions = candidate_positions[candidate_values < 0.2]
    keep[drop_positions] = False

    stats = {
        "raw_points": int(point_cloud.shape[0]),
        "high_points": int(high.sum()),
        "dropped_points": int(drop_positions.size),
        "kept_points": int(keep.sum()),
    }
    return point_cloud[keep], stats


def _metadata_value(metadata, key, batch_index):
    value = metadata[key]
    if isinstance(value, (list, tuple)):
        return value[batch_index]
    if torch.is_tensor(value):
        item = value[batch_index]
        return item.item() if item.numel() == 1 else item
    return value


def _zero_stats():
    return {
        "raw_points": 0,
        "high_points": 0,
        "dropped_points": 0,
        "kept_points": 0,
        "filtered_gt_pixels": 0,
    }


def _add_stats(total, batch_stats, gt_bev):
    for key in batch_stats:
        total[key] += batch_stats[key]
    total["filtered_gt_pixels"] += int(gt_bev.sum())


def build_filtered_lidar_gt(radar_cube, metadata, params, device=None):
    if device is None:
        device = radar_cube.device

    radar_np = radar_cube.detach().cpu().numpy()
    totals = _zero_stats()

    if radar_np.ndim != 4:
        raise ValueError(f"Expected single-frame radar batch with 4 dims, got {radar_np.shape}")

    gt_list = []
    for batch_index in range(radar_np.shape[0]):
        gt_path = _metadata_value(metadata, "gt_path", batch_index)
        raw_lidar = data_preparation.read_pointcloud(gt_path, mode="rs_lidar_clean")
        spar_map = compute_normalized_spar(radar_np[batch_index], params)
        filtered_lidar, stats = _filter_lidar_points(raw_lidar, spar_map, params)
        gt_bev = data_preparation.lidarpc_to_lidarcube(filtered_lidar, params) > 0
        gt_list.append(torch.from_numpy(gt_bev.astype(np.float32)))
        _add_stats(totals, stats, gt_bev)

    return torch.stack(gt_list, dim=0).to(device=device), totals


def build_filtered_lidar_gt_time(radar_cube, metadata, params, device=None):
    if device is None:
        device = radar_cube.device

    radar_np = radar_cube.detach().cpu().numpy()
    totals = _zero_stats()

    if radar_np.ndim != 5:
        raise ValueError(f"Expected temporal radar batch with 5 dims, got {radar_np.shape}")

    gt_batches = []
    for batch_index in range(radar_np.shape[0]):
        gt_frames = []
        for time_index in range(radar_np.shape[1]):
            frame_meta = metadata[time_index]
            gt_path = _metadata_value(frame_meta, "gt_path", batch_index)
            raw_lidar = data_preparation.read_pointcloud(gt_path, mode="rs_lidar_clean")
            spar_map = compute_normalized_spar(radar_np[batch_index, time_index], params)
            filtered_lidar, stats = _filter_lidar_points(raw_lidar, spar_map, params)
            gt_bev = data_preparation.lidarpc_to_lidarcube(filtered_lidar, params) > 0
            gt_frames.append(torch.from_numpy(gt_bev.astype(np.float32)))
            _add_stats(totals, stats, gt_bev)
        gt_batches.append(torch.stack(gt_frames, dim=0))

    return torch.stack(gt_batches, dim=0).to(device=device), totals
