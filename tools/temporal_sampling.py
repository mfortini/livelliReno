"""Causal temporal resampling shared contract with temporal-sampling.js.

Only observations with time <= cutoff are visible. Interpolated points never
extend past the last visible observation and gaps wider than max_gap stay
missing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MINUTE_MS = 60_000


def _as_points(times, values):
    t = np.asarray(times, dtype=np.int64)
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(t) & np.isfinite(v)
    t, v = t[ok], v[ok]
    order = np.argsort(t, kind="stable")
    return t[order], v[order]


def _visible(times, values, cutoff_ms, max_gap_minutes):
    t, v = _as_points(times, values)
    keep = t <= int(cutoff_ms)
    return t[keep], v[keep]


def pchip_slopes(t, y):
    n = len(t)
    if n < 2:
        return np.zeros(n)
    h = np.diff(t).astype(float)
    d = np.diff(y) / h
    m = np.zeros(n)
    if n == 2:
        m[:] = d[0]
        return m
    # endpoints (same one-sided rules as the JS implementation)
    m[0] = ((2 * h[0] + h[1]) * d[0] - h[0] * d[1]) / (h[0] + h[1])
    if m[0] * d[0] <= 0:
        m[0] = 0.0
    elif d[0] * d[1] < 0 and abs(m[0]) > abs(3 * d[0]):
        m[0] = 3 * d[0]
    m[-1] = ((2 * h[-1] + h[-2]) * d[-1] - h[-1] * d[-2]) / (h[-1] + h[-2])
    if m[-1] * d[-1] <= 0:
        m[-1] = 0.0
    elif d[-1] * d[-2] < 0 and abs(m[-1]) > abs(3 * d[-1]):
        m[-1] = 3 * d[-1]
    for i in range(1, n - 1):
        if d[i - 1] * d[i] <= 0:
            m[i] = 0.0
            continue
        w1 = 2 * h[i] + h[i - 1]
        w2 = h[i] + 2 * h[i - 1]
        m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    return m


def _segment_index(t, targets):
    """Return (left_idx, right_idx) for each target; right_idx may be n (past end)."""
    n = len(t)
    idx = np.searchsorted(t, targets, side="left")
    left = np.clip(idx - 1, -1, n - 1)
    right = np.clip(idx, 0, n - 1)
    exact = idx < n
    exact &= t[np.clip(idx, 0, n - 1)] == targets
    left = np.where(exact, idx, left)
    right = np.where(exact, idx, right)
    return left, right, exact


def sample_many(times, values, targets_ms, *, cutoff_ms, method="linear", max_gap_minutes=60):
    """Vectorized causal sample. Returns float array with NaN for missing."""
    targets = np.asarray(targets_ms, dtype=np.int64)
    out = np.full(targets.shape, np.nan, dtype=float)
    if targets.size == 0:
        return out
    t, v = _visible(times, values, cutoff_ms, max_gap_minutes)
    if t.size == 0:
        return out
    max_gap_ms = max_gap_minutes * MINUTE_MS
    # exact nodes first
    pos = np.searchsorted(t, targets)
    valid_pos = pos < t.size
    hit = np.zeros(targets.shape, dtype=bool)
    hit[valid_pos] = t[pos[valid_pos]] == targets[valid_pos]
    out[hit] = v[pos[hit]]

    need = ~hit
    if not need.any():
        return out
    tg = targets[need]
    idx = np.searchsorted(t, tg, side="left")
    has_left = idx > 0
    has_right = idx < t.size
    # targets before first or after last visible point stay missing
    ok = has_left & has_right
    # also allow target == first point handled by hit; target between points needs both
    li = np.clip(idx - 1, 0, t.size - 1)
    ri = np.clip(idx, 0, t.size - 1)
    lt, rt = t[li], t[ri]
    lv, rv = v[li], v[ri]
    gap_ok = (rt - lt) <= max_gap_ms
    inside = (tg >= lt) & (tg <= rt)
    good = ok & gap_ok & inside
    if not good.any():
        out[need] = np.nan
        return out
    tg_g = tg[good]
    lt_g, rt_g = lt[good], rt[good]
    lv_g, rv_g = lv[good], rv[good]
    w = (tg_g - lt_g) / (rt_g - lt_g)
    if method == "pchip" and t.size >= 2:
        slopes = pchip_slopes(t, v)
        li_g = li[good]
        ri_g = ri[good]
        h = (t[ri_g] - t[li_g]).astype(float)
        tt = (tg_g - t[li_g]) / np.where(h == 0, 1.0, h)
        m0 = slopes[li_g]
        m1 = slopes[ri_g]
        t2 = tt * tt
        t3 = t2 * tt
        vals = ((2 * t3 - 3 * t2 + 1) * v[li_g]
                + (t3 - 2 * t2 + tt) * h * m0
                + (-2 * t3 + 3 * t2) * v[ri_g]
                + (t3 - t2) * h * m1)
    else:
        vals = lv_g * (1 - w) + rv_g * w
    tmp = out[need]
    tmp[good] = vals
    out[need] = tmp
    return out


@dataclass
class TemporalSampler:
    times_ms: np.ndarray
    values: np.ndarray
    method: str = "linear"
    max_gap_minutes: float = 60.0

    @classmethod
    def from_frame(cls, df: pd.DataFrame, method="linear", max_gap_minutes=60.0):
        ts = pd.to_datetime(df["time"], utc=True)
        t = ((ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)).to_numpy(dtype=np.int64)
        v = df["value"].to_numpy(dtype=float)
        return cls(t, v, method, max_gap_minutes)

    def sample_many(self, targets_ms, cutoff_ms):
        return sample_many(
            self.times_ms, self.values, targets_ms,
            cutoff_ms=cutoff_ms, method=self.method,
            max_gap_minutes=self.max_gap_minutes,
        )
