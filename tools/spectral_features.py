#!/usr/bin/env python3
"""Causal spectral features shared with temporal-sampling.js.

Per-station features computed on the last W samples of the causal sampling
grid (newest-first input), identical math in JS and Python:

  - fill NaN by index linear interpolation (edges -> nearest finite);
  - subtract mean, apply Hann window w[i] = 0.5 - 0.5 cos(2*pi*i/(W-1));
  - direct DFT, bins j = 1..J (period W*step/j);
  - mag_j = hypot(re,im)/W; feat = log1p(mag_j);
  - power bands: centroid, dominant bin, low/mid/high power fractions;
  - Haar à trous wavelet, K scales: detail_k = a - avg2(a),
    a_next[i] = i >= 2^k ? (a[i] + a[i-2^k]) / 2 : a[i];
    feats: rms(detail_k), detail_k[W-1];
  - coverage = finite fraction of the input window.

Feature order (fixed): cov, fft1..J, centroid, dombin, bandlo, bandmid,
bandhi, wav{1..K}rms, wav{1..K}last. J=12, K=5 -> 28 features/station.
"""
import numpy as np

FFT_BINS = 12
WAVELET_SCALES = 5
N_SPECTRAL_FEATURES = 1 + FFT_BINS + 5 + 2 * WAVELET_SCALES  # 28


def spectral_feature_names(station_tag, fft_bins=FFT_BINS, wavelet_scales=WAVELET_SCALES):
    names = [f"spec_{station_tag}_cov"]
    names += [f"spec_{station_tag}_fft{j}" for j in range(1, fft_bins + 1)]
    names += [
        f"spec_{station_tag}_centroid",
        f"spec_{station_tag}_dombin",
        f"spec_{station_tag}_bandlo",
        f"spec_{station_tag}_bandmid",
        f"spec_{station_tag}_bandhi",
    ]
    names += [f"spec_{station_tag}_wav{k}rms" for k in range(1, wavelet_scales + 1)]
    names += [f"spec_{station_tag}_wav{k}last" for k in range(1, wavelet_scales + 1)]
    return names


def _fill_nan(values):
    """Index-linear interpolation; edges filled with nearest finite.
    Returns (filled array or None if <2 finite, finite_count)."""
    x = np.asarray(values, dtype=float)
    n = x.size
    finite = np.isfinite(x)
    count = int(finite.sum())
    if count < 2:
        return None, count
    idx = np.arange(n)
    if count == n:
        return x, count
    filled = np.interp(idx, idx[finite], x[finite])
    return filled, count


def _hann(n):
    if n <= 1:
        return np.ones(n)
    i = np.arange(n)
    return 0.5 - 0.5 * np.cos(2 * np.pi * i / (n - 1))


def _detrend(x):
    return x - np.mean(x)


def spectral_features(values_newest_first, fft_bins=FFT_BINS, wavelet_scales=WAVELET_SCALES):
    """Return a list of 28 floats for a newest-first window.

    Always returns finite numbers: degenerate windows produce zeros for the
    spectral block (coverage still reports data availability).
    """
    n = len(values_newest_first)
    zeros = [0.0] * (1 + fft_bins + 5 + 2 * wavelet_scales)
    if n < 4:
        zeros[0] = float(np.mean([np.isfinite(v) for v in values_newest_first])) if n else 0.0
        return zeros
    cov = float(np.mean(np.isfinite(np.asarray(values_newest_first, dtype=float))))
    # chronological order (oldest -> newest)
    filled, count = _fill_nan(values_newest_first[::-1])
    if filled is None:
        zeros[0] = cov
        return zeros
    x = _detrend(filled)
    w = _hann(n)
    xw = x * w
    # direct DFT for bins 1..fft_bins (angular freq 2*pi*j*i/n)
    i = np.arange(n)
    mags = []
    for j in range(1, fft_bins + 1):
        ang = 2 * np.pi * j * i / n
        re = float(np.sum(xw * np.cos(ang))) / n
        im = -float(np.sum(xw * np.sin(ang))) / n
        mags.append(float(np.hypot(re, im)))
    powers = np.asarray(mags, dtype=float) ** 2
    total = float(powers.sum())
    out = [cov]
    out += [float(np.log1p(m)) for m in mags]
    if total <= 0 or not np.isfinite(total):
        out += [0.0, 0.0, 0.0, 0.0, 0.0]
    else:
        js = np.arange(1, fft_bins + 1, dtype=float)
        centroid = float(np.sum(js * powers) / total)
        dombin = float(int(np.argmax(powers)) + 1)
        bandlo = float(powers[:3].sum() / total)
        bandmid = float(powers[3:8].sum() / total)
        bandhi = float(powers[8:].sum() / total)
        out += [centroid, dombin, bandlo, bandmid, bandhi]
    # Haar à trous wavelet on detrended (unwindowed) filled series
    a = x.copy()
    rms_feats = []
    last_feats = []
    for k in range(wavelet_scales):
        s = 1 << k
        a_next = a.copy()
        a_next[s:] = 0.5 * (a[s:] + a[:-s])
        detail = a - a_next
        rms_feats.append(float(np.sqrt(np.mean(detail * detail))))
        last_feats.append(float(detail[-1]))
        a = a_next
    out += rms_feats + last_feats
    return [float(v) if np.isfinite(v) else 0.0 for v in out]


def spectral_matrix(values_matrix_newest_first, fft_bins=FFT_BINS, wavelet_scales=WAVELET_SCALES):
    """Vectorized-over-rows wrapper: values (n_rows, W) newest-first with NaN.

    Returns (n_rows, N_SPECTRAL_FEATURES) float array.
    """
    n_rows, n = values_matrix_newest_first.shape
    out = np.zeros((n_rows, 1 + fft_bins + 5 + 2 * wavelet_scales), dtype=float)
    # Precompute trig tables for the DFT
    i = np.arange(n)
    if fft_bins > 0:
        COS = np.stack([np.cos(2 * np.pi * j * i / n) for j in range(1, fft_bins + 1)])  # (J, n)
        SIN = np.stack([np.sin(2 * np.pi * j * i / n) for j in range(1, fft_bins + 1)])
    else:
        COS = np.zeros((0, n))
        SIN = np.zeros((0, n))
    HANN = _hann(n)
    js = np.arange(1, fft_bins + 1, dtype=float)
    for r in range(n_rows):
        row = np.asarray(values_matrix_newest_first[r], dtype=float)
        finite = np.isfinite(row)
        cov = float(finite.mean())
        filled, count = _fill_nan(row[::-1])
        if filled is None or n < 4:
            out[r, 0] = cov
            continue
        x = _detrend(filled)
        xw = x * HANN
        # re/im via matrix multiply: (J, n) @ (n,) -> (J,)
        re = (COS @ xw) / n
        im = -(SIN @ xw) / n
        mags = np.hypot(re, im)
        powers = mags * mags
        total = float(powers.sum())
        out[r, 0] = cov
        out[r, 1:1 + fft_bins] = np.log1p(mags)
        col = 1 + fft_bins
        if total > 0 and np.isfinite(total):
            out[r, col] = float(np.sum(js * powers) / total)
            out[r, col + 1] = float(np.argmax(powers) + 1)
            out[r, col + 2] = float(powers[:3].sum() / total)
            out[r, col + 3] = float(powers[3:8].sum() / total)
            out[r, col + 4] = float(powers[8:].sum() / total)
        # Haar à trous
        a = x.copy()
        col = 1 + fft_bins + 5
        for k in range(wavelet_scales):
            s = 1 << k
            a_next = a.copy()
            a_next[s:] = 0.5 * (a[s:] + a[:-s])
            detail = a - a_next
            out[r, col + k] = float(np.sqrt(np.mean(detail * detail)))
            out[r, col + wavelet_scales + k] = float(detail[-1])
            a = a_next
    return out


if __name__ == "__main__":
    import sys
    rng = np.random.default_rng(0)
    # smoke test: white noise + sine
    t = np.arange(48)
    sig = 0.5 * np.sin(2 * np.pi * t / 12) + 0.05 * rng.standard_normal(48)
    sig[10] = np.nan
    feats = spectral_features(sig[::-1])
    print("cov", feats[0], "len", len(feats))
    print("dombin", feats[1 + FFT_BINS + 1], "centroid", feats[1 + FFT_BINS])
    m = spectral_matrix(np.stack([sig[::-1], sig[::-1]]))
    print("matrix parity:", np.allclose(m[0], np.asarray(feats), atol=1e-9))
    sys.exit(0)
