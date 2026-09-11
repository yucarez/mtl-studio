"""
Geometry conventions (used everywhere):

  theta (degrees) is the rotation of a text "frame" relative to the image x-axis,
  measured in image coordinates (y grows downward), so positive theta is a visual
  clockwise tilt. Frame axes:   u = (cos t, sin t)   (glyph "right")
                                n = (-sin t, cos t)  (glyph "down")
  A point (a, b) in a frame centred at c maps to image point  c + a*u + b*n.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def norm180(a: float) -> float:
    """Normalize angle to (-180, 180]."""
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


def norm90(a: float) -> float:
    """Normalize an *axis* angle (direction-less) to (-90, 90]."""
    a = norm180(a)
    if a > 90:
        a -= 180
    elif a <= -90:
        a += 180
    return a


def axes(theta: float) -> tuple[np.ndarray, np.ndarray]:
    t = math.radians(theta)
    return np.array([math.cos(t), math.sin(t)]), np.array([-math.sin(t), math.cos(t)])


def oriented_extent(points: np.ndarray, theta: float) -> tuple[tuple[float, float], float, float]:
    """Tight oriented rectangle (center, width, height) of points in a frame of angle theta."""
    u, n = axes(theta)
    p = np.asarray(points, np.float64).reshape(-1, 2)
    a, b = p @ u, p @ n
    a0, a1, b0, b1 = a.min(), a.max(), b.min(), b.max()
    ca, cb = (a0 + a1) / 2, (b0 + b1) / 2
    c = ca * u + cb * n
    return (float(c[0]), float(c[1])), float(a1 - a0), float(b1 - b0)


def rect_quad(center, w: float, h: float, theta: float) -> np.ndarray:
    u, n = axes(theta)
    c = np.asarray(center, np.float64)
    pts = [c - u * w / 2 - n * h / 2, c + u * w / 2 - n * h / 2,
           c + u * w / 2 + n * h / 2, c - u * w / 2 + n * h / 2]
    return np.array(pts, np.float32)


def frame_matrix(center, w: int, h: int, theta: float) -> np.ndarray:
    """Affine matrix mapping frame pixel (x, y) -> image pixel."""
    u, n = axes(theta)
    cx, cy = center
    return np.array([
        [u[0], n[0], cx - (w / 2) * u[0] - (h / 2) * n[0]],
        [u[1], n[1], cy - (w / 2) * u[1] - (h / 2) * n[1]],
    ], np.float64)


def rect_crop(img: np.ndarray, center, w: float, h: float, theta: float,
              interp=cv2.INTER_CUBIC, border=cv2.BORDER_REPLICATE, border_value=0) -> np.ndarray:
    """Extract an upright crop of an oriented rectangle."""
    w, h = max(1, int(round(w))), max(1, int(round(h)))
    M = frame_matrix(center, w, h, theta)
    return cv2.warpAffine(img, M, (w, h), flags=interp | cv2.WARP_INVERSE_MAP,
                          borderMode=border, borderValue=border_value)


def paste_frame(dst: np.ndarray, patch_rgba: np.ndarray, center, theta: float) -> tuple[int, int, int, int]:
    """Alpha-composite an RGBA frame patch onto an RGB uint8 image in-place.
    Returns the affected bbox (x0, y0, x1, y1)."""
    h, w = patch_rgba.shape[:2]
    M = frame_matrix(center, w, h, theta)
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], np.float64) @ M.T
    x0 = max(0, int(math.floor(corners[:, 0].min())) - 1)
    y0 = max(0, int(math.floor(corners[:, 1].min())) - 1)
    x1 = min(dst.shape[1], int(math.ceil(corners[:, 0].max())) + 1)
    y1 = min(dst.shape[0], int(math.ceil(corners[:, 1].max())) + 1)
    if x1 <= x0 or y1 <= y0:
        return (0, 0, 0, 0)
    M2 = M.copy()
    M2[0, 2] -= x0
    M2[1, 2] -= y0
    if abs(theta) < 1e-6 and float(center[0] - w / 2).is_integer() and float(center[1] - h / 2).is_integer():
        interp = cv2.INTER_NEAREST     # exact placement: no resampling blur
    else:
        interp = cv2.INTER_LINEAR
    warped = cv2.warpAffine(patch_rgba, M2, (x1 - x0, y1 - y0), flags=interp,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    alpha = warped[..., 3:4].astype(np.float32) / 255.0
    roi = dst[y0:y1, x0:x1].astype(np.float32)
    roi = roi * (1 - alpha) + warped[..., :3].astype(np.float32) * alpha
    dst[y0:y1, x0:x1] = np.clip(roi + 0.5, 0, 255).astype(np.uint8)
    return (x0, y0, x1, y1)


def warp_mask_to_frame(mask: np.ndarray, center, w: int, h: int, theta: float) -> np.ndarray:
    """Image-space bool mask -> frame-space bool mask."""
    m = rect_crop(mask.astype(np.uint8) * 255, center, w, h, theta, interp=cv2.INTER_NEAREST,
                  border=cv2.BORDER_CONSTANT, border_value=0)
    return m > 127


def frame_mask_to_image(mask: np.ndarray, center, theta: float, shape) -> np.ndarray:
    h, w = mask.shape
    M = frame_matrix(center, w, h, theta)
    out = cv2.warpAffine(mask.astype(np.uint8) * 255, M, (shape[1], shape[0]), flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out > 127


def polygon_mask(shape, quad: np.ndarray) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(quad).astype(np.int32)], 1)
    return m.astype(bool)


def bbox_iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def bbox_overlap_small(a, b) -> float:
    """Intersection over the smaller box's area."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / small if small > 0 else 0.0


def quad_bbox(quad: np.ndarray) -> tuple[float, float, float, float]:
    return (float(quad[:, 0].min()), float(quad[:, 1].min()),
            float(quad[:, 0].max()), float(quad[:, 1].max()))
