import numpy as np
import torch

HEATMAPS_RESOLUTION = 64

def raise_left_eyebrow(landmarks, fig_size, factor=1):
    if isinstance(fig_size, tuple):
        fig_size = fig_size[0]
    idxs = np.array([17, 18, 19, 20, 21])
    scale = fig_size * factor / 600
    shifts = np.array([12, 10, 8, 7, 6]) * scale
    landmarks[idxs, 1] -= shifts.astype(int)
    return landmarks


def raise_right_eyebrow(landmarks, fig_size, factor=1):
    if isinstance(fig_size, tuple):
        fig_size = fig_size[0]
    idxs = np.array([26, 25, 24, 23, 22])
    scale = fig_size * factor / 600
    shifts = np.array([12, 10, 8, 7, 6]) * scale
    print(shifts)
    landmarks[idxs, 1] -= shifts.astype(int)
    return landmarks


def raise_eyebrows(landmarks, fig_size, factor=1):
    landmarks = raise_right_eyebrow(landmarks, fig_size, factor)
    landmarks = raise_left_eyebrow(landmarks, fig_size, factor)
    return landmarks


def smile(landmarks, fig_size, factor=1):
    idxs = np.arange(48, 68).astype(int)
    scale = fig_size * factor / 900
    x_shifts = np.array([-5,-5,-3,-2,-3,-5,7,  -2,1,-1,1,-2,  -7,-4,-4,-4,-1,  0,0,0]) * scale
    y_shifts = np.array([-5,-5,-1,0,1,5,7,     5,3,0,-3,-5,   -7,-3,0,3,7,  -1,0,1]) * scale
    landmarks[idxs, 0] += y_shifts.astype(int)
    landmarks[idxs, 1] += x_shifts.astype(int)
    return landmarks


def raise_left_eyebrow_hm(heatmaps, factor=1):
    idxs = np.array([17, 18, 19, 20, 21])
    scale = HEATMAPS_RESOLUTION * factor / 400
    shifts = np.array([12, 10, 8, 7, 6]) * scale
    for i, idx in enumerate(idxs):
        shift = int(shifts[i])
        y = np.arange(64-shift)
        heatmaps.data[idx, y, :] = heatmaps.data[idx, y + shift, :]
        heatmaps.data[idx, -shift:, :] = 0
    return heatmaps


def raise_right_eyebrow_hm(heatmaps, factor=1):
    idxs = np.array([26, 25, 24, 23, 22])
    scale = HEATMAPS_RESOLUTION * factor / 400
    shifts = np.array([12, 10, 8, 7, 6]) * scale
    for i, idx in enumerate(idxs):
        shift = int(shifts[i])
        y = np.arange(64-shift)
        heatmaps.data[idx, y, :] = heatmaps.data[idx, y + shift, :]
        heatmaps.data[idx, -shift:, :] = 0
    return heatmaps


def raise_eyebrows_hm(heatmaps, factor=1):
    heatmaps = raise_right_eyebrow_hm(heatmaps, factor)
    heatmaps = raise_left_eyebrow_hm(heatmaps, factor)
    return heatmaps


def smile_hm(heatmaps, factor=1):
    idxs = torch.arange(48, 68)
    scale = HEATMAPS_RESOLUTION * factor / 900
    x_shifts = torch.tensor([-5,-5,-3,-2,-3,-5,7,  -2,1,-1,1,-2,  -7,-4,-4,-4,-1,  0,0,0]) * scale
    y_shifts = torch.tensor([-5,-5,-1,0,1,5,7,     5,3,0,-3,-5,   -7,-3,0,3,7,  -1,0,1]) * scale
    for i, idx in enumerate(idxs):
        x_shift = int(x_shifts[i])
        y_shift = int(y_shifts[i])
        y = np.arange(64 - y_shift)
        x = np.arange(64 - x_shift)
        heatmaps.data[idx, y, :] += heatmaps.data[idx, y + y_shift, :]
        heatmaps.data[idx, -y_shift, :] = 0
        heatmaps.data[idx, :, x] += heatmaps.data[idx, :, x + x_shift]
        heatmaps.data[idx, :, -x_shift] = 0
    return heatmaps
