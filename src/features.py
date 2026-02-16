import cv2
import numpy as np
from scipy.stats import kurtosis, skew

EPS = 1e-6


def _normalize_u8(arr):
    mn, mx = arr.min(), arr.max()
    if mx - mn < EPS:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)


def _glcm(img_u8, distances=(1, 2, 4), angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)):
    feats = []
    h, w = img_u8.shape
    nl = 16
    q = np.clip(img_u8 // 16, 0, nl - 1).astype(np.int32)
    ii, jj = np.meshgrid(np.arange(nl), np.arange(nl), indexing="ij")

    for d in distances:
        for a in angles:
            dy, dx = int(round(d * np.sin(a))), int(round(d * np.cos(a)))
            y1, y2 = max(0, dy), min(h, h + dy)
            x1, x2 = max(0, dx), min(w, w + dx)
            y1s, y2s = max(0, -dy), min(h, h - dy)
            x1s, x2s = max(0, -dx), min(w, w - dx)
            p1 = q[y1:y2, x1:x2].ravel()
            p2 = q[y1s:y2s, x1s:x2s].ravel()

            g = np.zeros((nl, nl), dtype=np.float64)
            np.add.at(g, (p1, p2), 1)
            g /= g.sum() + EPS

            mi, mj = np.sum(ii * g), np.sum(jj * g)
            si = np.sqrt(np.sum((ii - mi) ** 2 * g))
            sj = np.sqrt(np.sum((jj - mj) ** 2 * g))
            corr = (
                np.sum((ii - mi) * (jj - mj) * g) / (si * sj)
                if si > 0 and sj > 0
                else 0.0
            )

            feats.extend(
                [
                    np.sum((ii - jj) ** 2 * g),
                    np.sum(g / (1 + np.abs(ii - jj))),
                    np.sum(g**2),
                    corr,
                    -np.sum(g[g > 0] * np.log2(g[g > 0])),
                    np.sum(np.abs(ii - jj) * g),
                ]
            )
    return feats


def _lbp(img_u8, radius=1, n_points=8, n_bins=32):
    h, w = img_u8.shape
    lbp = np.zeros((h - 2 * radius, w - 2 * radius), dtype=np.int32)
    for n in range(n_points):
        a = 2 * np.pi * n / n_points
        dy, dx = int(round(radius * np.sin(a))), int(round(radius * np.cos(a)))
        c = img_u8[radius : h - radius, radius : w - radius].astype(np.float32)
        nb = img_u8[
            radius + dy : h - radius + dy, radius + dx : w - radius + dx
        ].astype(np.float32)
        lbp += (nb >= c).astype(np.int32) << n
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, 2**n_points))
    return (hist / (hist.sum() + EPS)).tolist()


def _band_stats(band):
    std_val = band.std()
    return [
        band.mean(),
        std_val,
        *[np.percentile(band, p) for p in (5, 10, 25, 50, 75, 90, 95)],
        skew(band.ravel()) if std_val > 1e-8 else 0.0,
        kurtosis(band.ravel()) if std_val > 1e-8 else 0.0,
    ]


def _index_stats(idx):
    std_val = idx.std()
    return [
        idx.mean(),
        std_val,
        *[np.percentile(idx, p) for p in (5, 10, 25, 50, 75, 90, 95)],
        skew(idx.ravel()) if std_val > 1e-8 else 0.0,
        kurtosis(idx.ravel()) if std_val > 1e-8 else 0.0,
        idx.max() - idx.min(),
    ]


def _array_stats(arr):
    std_val = arr.std()
    return [
        arr.mean(),
        std_val,
        *[np.percentile(arr, p) for p in (10, 25, 50, 75, 90)],
        skew(arr) if std_val > 1e-8 else 0.0,
        kurtosis(arr) if std_val > 1e-8 else 0.0,
        arr.max() - arr.min(),
    ]


def _pixel_entropy(vals, n_bins=32):
    h, _ = np.histogram(vals, bins=n_bins)
    p = h / (h.sum() + 1e-8)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def extract(sample) -> np.ndarray:
    hs, ms, rgb = sample["hs"], sample["ms"], sample["rgb"]
    feats = []

    # MS features
    if ms is not None:
        blue_ms = ms[:, :, 0]
        nir, red, re, green = ms[:, :, 4], ms[:, :, 2], ms[:, :, 3], ms[:, :, 1]
        ndvi = (nir - red) / (nir + red + EPS)
        ndvi_u8, nir_u8, re_u8 = (
            _normalize_u8(ndvi),
            _normalize_u8(nir),
            _normalize_u8(re),
        )
        for img in (ndvi_u8, nir_u8, re_u8):
            feats.extend(_glcm(img, distances=(1,), angles=(0, np.pi / 2)))
            feats.extend(_lbp(img, 1, 8))

        h, w = ndvi.shape
        quads = [
            ndvi[: h // 2, : w // 2],
            ndvi[: h // 2, w // 2 :],
            ndvi[h // 2 :, : w // 2],
            ndvi[h // 2 :, w // 2 :],
        ]
        qm = [q.mean() for q in quads]
        qs = [q.std() for q in quads]
        feats.extend(
            [np.std(qm), np.max(qm) - np.min(qm), np.std(qs), np.max(qs) - np.min(qs)]
        )

        bh, bw = h // 4, w // 4
        bm, bst = [], []
        for bi in range(4):
            for bj in range(4):
                blk = ndvi[bi * bh : (bi + 1) * bh, bj * bw : (bj + 1) * bw]
                bm.append(blk.mean())
                bst.append(blk.std())
        feats.extend([np.std(bm), np.max(bm) - np.min(bm), np.mean(bst), np.std(bst)])

        ndre = (nir - re) / (nir + re + EPS)
        gndvi = (nir - green) / (nir + green + EPS)
        ci_re = nir / (re + EPS) - 1.0
        ci_green = nir / (green + EPS) - 1.0
        savi = 1.5 * (nir - red) / (nir + red + 0.5 + EPS)
        for idx in (ndvi, ndre, gndvi, ci_re, ci_green, savi):
            feats.extend(_index_stats(idx))

        for band in (blue_ms, green, red, re, nir):
            feats.extend(_band_stats(band))
    else:
        feats.extend([0] * 267)

    # RGB features
    if rgb is not None:
        for c in range(3):
            ch = rgb[:, :, c]
            feats.extend(
                [
                    ch.mean(),
                    ch.std(),
                    *[np.percentile(ch, p) for p in (10, 25, 50, 75, 90)],
                ]
            )
        hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        for c in range(3):
            feats.extend(
                [
                    hsv[:, :, c].mean(),
                    hsv[:, :, c].std(),
                    np.percentile(hsv[:, :, c], 50),
                ]
            )
        gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        feats.extend(_glcm(gray, distances=(1,), angles=(0, np.pi / 2)))
        feats.extend(_lbp(gray))
        gy, gx = np.gradient(rgb.mean(axis=2))
        grad = np.sqrt(gx**2 + gy**2)
        feats.extend([grad.mean(), grad.std(), grad.max()])
    else:
        feats.extend([0] * 77)

    # HS features
    pixels = hs.reshape(-1, 125)
    band_means = pixels.mean(axis=0)
    band_stds = pixels.std(axis=0)
    clean_slice = slice(10, 111)
    clean_means = band_means[clean_slice]
    clean_pixels = pixels[:, clean_slice]

    feats.extend(
        [
            clean_means.sum(),
            clean_means[:30].sum(),
            clean_means[30:60].sum(),
            clean_means[60:].sum(),
        ]
    )
    total = clean_means.sum() + EPS
    feats.extend(
        [
            clean_means[:30].sum() / total,
            clean_means[30:60].sum() / total,
            clean_means[60:].sum() / total,
        ]
    )
    clean_medians = np.median(clean_pixels, axis=0)
    med_total = clean_medians.sum() + EPS
    feats.extend(
        [
            clean_medians[:30].sum() / med_total,
            clean_medians[30:60].sum() / med_total,
            clean_medians[60:].sum() / med_total,
        ]
    )
    for s, e in [(0, 30), (30, 60), (60, 101)]:
        feats.append(clean_means[s:e].mean() - clean_medians[s:e].mean())

    hull_red = np.interp(range(20, 50), [20, 50], [clean_means[10], clean_means[40]])
    cr_red = clean_means[10:40] / (hull_red + EPS)
    feats.extend([cr_red.min(), cr_red.mean(), cr_red.std(), np.argmin(cr_red)])

    d1 = np.diff(clean_means)
    d2 = np.diff(d1)
    feats.extend(_array_stats(d1))
    feats.extend(_array_stats(d2))
    d1_diag = [45, 50, 55, 58, 60, 62, 65, 70]
    for b in d1_diag:
        if b < len(d1):
            feats.append(d1[b])
    feats.extend([d1.max(), d1.argmax(), d2.max(), d2.argmax()])
    px_d1 = np.diff(clean_pixels, axis=1)
    px_d1_std = px_d1.std(axis=0)
    feats.extend([px_d1_std.mean(), px_d1_std.std(), px_d1_std.max()])

    # HS spatial heterogeneity
    h, w = hs.shape[:2]
    quad_spectra = [
        hs[: h // 2, : w // 2].reshape(-1, 125)[:, clean_slice].mean(0),
        hs[: h // 2, w // 2 :].reshape(-1, 125)[:, clean_slice].mean(0),
        hs[h // 2 :, : w // 2].reshape(-1, 125)[:, clean_slice].mean(0),
        hs[h // 2 :, w // 2 :].reshape(-1, 125)[:, clean_slice].mean(0),
    ]
    quad_arr = np.array(quad_spectra)
    quad_cv = quad_arr.std(axis=0) / (quad_arr.mean(axis=0) + EPS)
    feats.extend([quad_cv.mean(), quad_cv.std(), quad_cv.max()])
    quad_ndvi = []
    for qs in quad_spectra:
        qr, qn = qs[20:35].mean(), qs[70:90].mean()
        quad_ndvi.append((qn - qr) / (qn + qr + EPS))
    feats.extend([np.std(quad_ndvi), np.max(quad_ndvi) - np.min(quad_ndvi)])

    nir_px = pixels[:, 80:100].mean(1)
    nir_med = np.median(nir_px)
    above = nir_px[nir_px >= nir_med]
    below = nir_px[nir_px < nir_med]
    feats.append(above.mean() / (below.mean() + EPS) if len(below) > 0 else 1.0)
    feats.append(kurtosis(nir_px) if nir_px.std() > EPS else 0.0)

    px_brightness = pixels[:, 10:110].mean(1)
    bright_mask = px_brightness >= np.percentile(px_brightness, 75)
    dark_mask = px_brightness <= np.percentile(px_brightness, 25)
    bright_spec = pixels[bright_mask][:, 10:110].mean(0)
    dark_spec = pixels[dark_mask][:, 10:110].mean(0)
    cos_sim = np.dot(bright_spec, dark_spec) / (
        np.linalg.norm(bright_spec) * np.linalg.norm(dark_spec) + EPS
    )
    feats.append(cos_sim)

    # HS per-band pixel distributions
    diag_bands = [44, 54, 60, 70, 91, 99]
    for b in diag_bands:
        band_px = pixels[:, b]
        std_b = band_px.std()
        feats.extend(
            [
                _pixel_entropy(band_px),
                skew(band_px) if std_b > EPS else 0.0,
                kurtosis(band_px) if std_b > EPS else 0.0,
                np.percentile(band_px, 75) - np.percentile(band_px, 25),
            ]
        )

    red_hs = pixels[:, 30:45].mean(1)
    re_hs = pixels[:, 50:60].mean(1)
    nir_hs = pixels[:, 80:100].mean(1)
    ndvi_hs = (nir_hs - red_hs) / (nir_hs + red_hs + EPS)
    ndre_hs = (nir_hs - re_hs) / (nir_hs + re_hs + EPS)
    red_nir = red_hs / (nir_hs + EPS)
    re_deriv = np.diff(pixels[:, 45:75], axis=1)
    reip = re_deriv.argmax(1).astype(np.float32)
    reip_val = re_deriv.max(1)

    for arr in (ndvi_hs, ndre_hs, red_nir, reip, reip_val):
        feats.extend(_array_stats(arr))

    vis, re_r, nir_r = (
        band_means[10:40].mean(),
        band_means[40:65].mean(),
        band_means[75:110].mean(),
    )
    feats.extend([vis / (nir_r + EPS), re_r / (nir_r + EPS), vis / (re_r + EPS)])

    pri = (band_means[20] - band_means[30]) / (band_means[20] + band_means[30] + EPS)
    ari = 1.0 / (band_means[25] + EPS) - 1.0 / (band_means[62] + EPS)
    cri = 1.0 / (band_means[18] + EPS) - 1.0 / (band_means[28] + EPS)
    feats.extend([pri, ari, cri])

    key_bands = [12, 22, 32, 42, 52, 62, 72, 82, 92, 102]
    for i in range(len(key_bands)):
        for j in range(i + 1, len(key_bands)):
            bi, bj = key_bands[i], key_bands[j]
            feats.append(
                (band_means[bi] - band_means[bj])
                / (band_means[bi] + band_means[bj] + EPS)
            )

    for start, end in [(25, 35), (35, 45), (50, 60), (75, 85), (85, 95), (95, 105)]:
        window = pixels[:, start:end].mean(1)
        feats.extend([window.mean(), window.std(), skew(window), kurtosis(window)])

    for s, e in [(10, 30), (30, 50), (50, 70), (70, 95), (95, 110)]:
        region = band_means[s:e]
        feats.extend([region.mean(), region.std()])

    # HS discriminative spectral features
    nir_plateau = band_means[80:105].mean()
    feats.append(nir_plateau)

    green_region = band_means[20:30].mean()
    green_neighbors = (band_means[12:18].mean() + band_means[32:38].mean()) / 2
    green_prominence = green_region - green_neighbors
    feats.extend([green_prominence, green_region / (green_neighbors + EPS)])

    re_slope = (band_means[75] - band_means[60]) / 15.0
    feats.append(re_slope)

    re_derivs = np.diff(band_means[55:80])
    re_inflection = np.argmax(re_derivs) + 55
    feats.append(float(re_inflection))

    vis_level = band_means[30:55].mean()
    vis_nir_contrast = vis_level - nir_plateau
    vis_nir_ratio = vis_level / (nir_plateau + EPS)
    feats.extend([vis_nir_contrast, vis_nir_ratio])

    nir_early = band_means[80:90].mean()
    nir_late = band_means[95:105].mean()
    nir_slope = nir_late - nir_early
    feats.append(nir_slope)

    zero_mask = np.all(hs < 1, axis=2)
    feats.append(zero_mask.sum() / (hs.shape[0] * hs.shape[1]))

    # HS texture on critical bands
    re_spatial = _normalize_u8(hs[:, :, 43:59].mean(axis=2))
    nir_spatial = _normalize_u8(hs[:, :, 80:95].mean(axis=2))
    for img in (re_spatial, nir_spatial):
        feats.extend(_glcm(img, distances=(1,), angles=(0, np.pi / 2)))
        feats.extend(_lbp(img, 1, 8))

    nir_region = hs[:, :, 80:100].mean(axis=2)
    feats.extend([nir_region.std(), nir_region.mean() / (nir_region.std() + EPS)])

    # HS consistency (CV per band / pixel)
    cv_per_band = band_stds[10:110] / (band_means[10:110] + EPS)
    feats.extend(
        [
            cv_per_band.mean(),
            cv_per_band.std(),
            cv_per_band.min(),
            cv_per_band.max(),
            np.percentile(cv_per_band, 25),
            np.percentile(cv_per_band, 75),
        ]
    )

    pixel_means = pixels[:, 10:110].mean(axis=1)
    pixel_stds = pixels[:, 10:110].std(axis=1)
    pixel_cv = pixel_stds / (pixel_means + EPS)
    feats.extend(
        [
            pixel_cv.mean(),
            pixel_cv.std(),
            np.percentile(pixel_cv, 10),
            np.percentile(pixel_cv, 90),
        ]
    )

    nir_cv = band_stds[80:100] / (band_means[80:100] + EPS)
    feats.extend([nir_cv.mean(), nir_cv.std()])

    re_cv = band_stds[50:65] / (band_means[50:65] + EPS)
    feats.extend([re_cv.mean(), re_cv.std()])

    # HS red-edge onset (Mahlein et al., Plant Methods 2012)
    re_onset = band_means[44:60]
    re_onset_px = pixels[:, 44:60]
    feats.extend(
        [
            re_onset.mean(),
            re_onset.min(),
            re_onset.max() - re_onset.min(),
            re_onset.std(),
        ]
    )
    nir_mean = band_means[80:100].mean()
    feats.extend(
        [
            re_onset.mean() / (nir_mean + EPS),
            re_onset.min() / (nir_mean + EPS),
        ]
    )
    re_onset_px_mean = re_onset_px.mean(1)
    feats.extend(
        [
            re_onset_px_mean.std(),
            skew(re_onset_px_mean) if re_onset_px_mean.std() > EPS else 0.0,
            kurtosis(re_onset_px_mean) if re_onset_px_mean.std() > EPS else 0.0,
        ]
    )

    blue_region = band_means[10:14]
    feats.extend(
        [
            blue_region.mean(),
            blue_region.mean() / (nir_mean + EPS),
            blue_region.mean() / (re_onset.mean() + EPS),
        ]
    )

    # Red absorption depth — deeper = more chlorophyll-b damage from rust
    green_px = pixels[:, 20:30].mean(1)
    red_px = pixels[:, 35:45].mean(1)
    nir_px_depth = pixels[:, 80:95].mean(1)
    red_depth = ((green_px + nir_px_depth) / 2 - red_px) / (
        (green_px + nir_px_depth) / 2 + EPS
    )
    feats.extend(
        [
            red_depth.mean(),
            red_depth.std(),
            np.percentile(red_depth, 25),
            np.percentile(red_depth, 75),
        ]
    )

    re_transition = band_means[48:65]
    d1_re = np.diff(re_transition)
    d2_re = np.diff(d1_re)
    feats.extend(
        [
            d1_re.max(),
            d1_re.argmax(),
            d2_re.max(),
            d2_re.min(),
            d2_re.std(),
        ]
    )

    # Cross-modal features
    if ms is not None:
        nir_ms_m = ms[:, :, 4].mean()
        red_ms_m = ms[:, :, 2].mean()
        ndvi_ms_v = (nir_ms_m - red_ms_m) / (nir_ms_m + red_ms_m + EPS)

        nir_hs_m = band_means[80:100].mean()
        red_hs_m = band_means[30:45].mean()
        ndvi_hs_v = (nir_hs_m - red_hs_m) / (nir_hs_m + red_hs_m + EPS)

        feats.append(ndvi_ms_v - ndvi_hs_v)
        feats.append(ndvi_ms_v * ndvi_hs_v)

        hs_ranges = [(12, 22), (22, 32), (38, 48), (55, 65), (80, 100)]  # clean bands
        for band_idx in range(5):
            ms_std = ms[:, :, band_idx].std()
            hs_spatial = hs[:, :, hs_ranges[band_idx][0] : hs_ranges[band_idx][1]].mean(
                2
            )
            hs_std_v = hs_spatial.std()
            feats.append(ms_std / (hs_std_v + EPS))
            feats.append(ms_std - hs_std_v)
    else:
        feats.extend([0] * 12)

    if rgb is not None:
        gray_cm = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        rgb_contrast = gray_cm.std()
        hs_contrast = hs.std(axis=(0, 1)).mean()
        feats.append(rgb_contrast / (hs_contrast + EPS))
    else:
        feats.append(0)

    return np.array(feats, dtype=np.float32)


def extract_batch(samples) -> np.ndarray:
    X = np.array([extract(s) for s in samples])
    return np.nan_to_num(X, 0, posinf=0, neginf=0)
