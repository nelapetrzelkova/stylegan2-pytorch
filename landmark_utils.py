import numpy as np
import torch

HEATMAPS_RESOLUTION = 64
LEFT_EYEBROW_IDXS = np.array([17, 18, 19, 20, 21])
RIGHT_EYEBROW_IDXS = np.array([26, 25, 24, 23, 22])
BOTH_EYEBROWS_IDXS = np.concatenate((LEFT_EYEBROW_IDXS, RIGHT_EYEBROW_IDXS))
MOUTH_IDXS = np.arange(48, 68).astype(int)
TRANSFORM_IDXS_DICT = {'smile': MOUTH_IDXS, 'raise_left_eyebrow': LEFT_EYEBROW_IDXS, 'raise_right_eyebrow': RIGHT_EYEBROW_IDXS, 'raise_eyebrows': BOTH_EYEBROWS_IDXS}

def raise_left_eyebrow(landmarks, fig_size, factor=1):
    if isinstance(fig_size, tuple):
        fig_size = fig_size[0]
    scale = fig_size * factor / 600
    shifts = np.array([12, 10, 8, 7, 6]) * scale
    landmarks[LEFT_EYEBROW_IDXS, 1] -= shifts.astype(int)
    return landmarks


def raise_right_eyebrow(landmarks, fig_size, factor=1):
    if isinstance(fig_size, tuple):
        fig_size = fig_size[0]
    scale = fig_size * factor / 600
    shifts = np.array([12, 10, 8, 7, 6]) * scale
    landmarks[RIGHT_EYEBROW_IDXS, 1] -= shifts.astype(int)
    return landmarks


def raise_eyebrows(landmarks, fig_size, factor=1):
    landmarks = raise_right_eyebrow(landmarks, fig_size, factor)
    landmarks = raise_left_eyebrow(landmarks, fig_size, factor)
    return landmarks


def smile(landmarks, fig_size, factor=1):
    scale = fig_size * factor / 900
    x_shifts = np.array([-5,-5,-3,-2,-3,-5,7,  -2,1,-1,1,-2,  -7,-4,-4,-4,-1,  0,0,0]) * scale
    y_shifts = np.array([-5,-5,-1,0,1,5,7,     5,3,0,-3,-5,   -7,-3,0,3,7,  -1,0,1]) * scale
    landmarks[MOUTH_IDXS, 0] += y_shifts.astype(int)
    landmarks[MOUTH_IDXS, 1] += x_shifts.astype(int)
    return landmarks


def raise_left_eyebrow_hm(heatmaps, factor=1):
    scale = HEATMAPS_RESOLUTION * factor / 400
    shifts = np.array([12, 10, 8, 7, 6]) * scale
    for i, idx in enumerate(LEFT_EYEBROW_IDXS):
        shift = int(shifts[i])
        y = np.arange(64-shift)
        heatmaps.data[idx, y, :] = heatmaps.data[idx, y + shift, :]
        heatmaps.data[idx, -shift:, :] = 0
    return heatmaps


def raise_right_eyebrow_hm(heatmaps, factor=1):
    scale = HEATMAPS_RESOLUTION * factor / 400
    shifts = np.array([12, 10, 8, 7, 6]) * scale
    for i, idx in enumerate(RIGHT_EYEBROW_IDXS):
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
    scale = HEATMAPS_RESOLUTION * factor / 900
    x_shifts = torch.tensor([-5,-5,-3,-2,-3,-5,7,  -2,1,-1,1,-2,  -7,-4,-4,-4,-1,  0,0,0]) * scale
    y_shifts = torch.tensor([-5,-5,-1,0,1,5,7,     5,3,0,-3,-5,   -7,-3,0,3,7,  -1,0,1]) * scale
    for i, idx in enumerate(MOUTH_IDXS):
        x_shift = int(x_shifts[i])
        y_shift = int(y_shifts[i])
        y = np.arange(HEATMAPS_RESOLUTION - y_shift)
        x = np.arange(HEATMAPS_RESOLUTION - x_shift)
        heatmaps.data[idx, y, :] += heatmaps.data[idx, y + y_shift, :]
        heatmaps.data[idx, -y_shift, :] = 0
        heatmaps.data[idx, :, x] += heatmaps.data[idx, :, x + x_shift]
        heatmaps.data[idx, :, -x_shift] = 0
    return heatmaps


def mask_image(img, heatmaps, augmentation):
    idxs = TRANSFORM_IDXS_DICT[augmentation]
    for idx in idxs:
        peak = torch.argmax(heatmaps[idx, :, :])
        coords = np.unravel_index(peak.detach().cpu().numpy(), (HEATMAPS_RESOLUTION, HEATMAPS_RESOLUTION))

