import numpy as np


def raise_left_eyebrow(landmarks, fig_size, factor=1):
    if isinstance(fig_size, tuple):
        fig_size = fig_size[0]
    idxs = [17, 18, 19, 20, 21]
    shifts = np.array([12, 10, 8, 7, 6])/600*fig_size * factor
    landmarks[idxs, 1] -= shifts.astype(int)
    return landmarks

def raise_right_eyebrow(landmarks, fig_size, factor=1):
    if isinstance(fig_size, tuple):
        fig_size = fig_size[0]
    idxs = [26, 25, 24, 23, 22]
    shifts = np.array([12, 10, 8, 7, 6])/600*fig_size * factor
    landmarks[idxs, 1] -= shifts.astype(int)
    return landmarks


def raise_eyebrows(landmarks, fig_size, factor=1):
    landmarks = raise_right_eyebrow(landmarks, fig_size, factor)
    landmarks = raise_left_eyebrow(landmarks, fig_size, factor)
    return landmarks


def smile(landmarks, fig_size, factor=1):
    idxs = np.arange(48, 68)
    scale = fig_size * factor / 900
    x_shifts = np.array([-5,-5,-3,-2,-3,-5,7,  -2,1,-1,1,-2,  -7,-4,-4,-4,-1,  0,0,0]) * scale
    y_shifts = np.array([-5,-5,-1,0,1,5,7,     5,3,0,-3,-5,   -7,-3,0,3,7,  -1,0,1]) * scale
    landmarks[idxs, 0] += y_shifts.astype(int)
    landmarks[idxs, 1] += x_shifts.astype(int)
    return landmarks