import cv2
import numpy as np
from scipy.stats import kurtosis, skew

EPS = 1e-6


def _normalize_u8(arr):
    mn, mx = arr.min(), arr.max()
    if mx - mn < EPS:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)


# ------------------------------------------------------------------ GLCM
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


# ------------------------------------------------------------------ LBP
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


# ------------------------------------------------------------------ Gabor
def _gabor(
    img_u8, frequencies=(0.1, 0.2, 0.4), thetas=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
):
    feats = []
    img_f = img_u8.astype(np.float32) / 255.0
    for freq in frequencies:
        for theta in thetas:
            sigma = 1.0 / (freq * np.pi)
            ks = max(3, min(int(sigma * 6) | 1, 31))
            kernel = cv2.getGaborKernel(
                (ks, ks), sigma, theta, 1.0 / freq, 0.5, 0, ktype=cv2.CV_32F
            )
            filt = cv2.filter2D(img_f, cv2.CV_32F, kernel)
            feats.extend([filt.mean(), filt.std(), np.abs(filt).mean()])
    return feats


# ------------------------------------------------------------------ Stats helpers
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


# ================================================================== MAIN
def extract(sample) -> np.ndarray:
    hs, ms, rgb = sample["hs"], sample["ms"], sample["rgb"]
    feats = []

    # ======================== MS (texture only - spectral redundant with HS) ========================
    if ms is not None:
        nir, red, re, green = ms[:, :, 4], ms[:, :, 2], ms[:, :, 3], ms[:, :, 1]

        # MS has 2x resolution (64x64 vs HS 32x32) - use for texture only
        ndvi = (nir - red) / (nir + red + EPS)

        # Texture: GLCM + LBP + Gabor on NDVI, NIR, RedEdge (high-res 64x64)
        ndvi_u8, nir_u8, re_u8 = (
            _normalize_u8(ndvi),
            _normalize_u8(nir),
            _normalize_u8(re),
        )
        for img in (ndvi_u8, nir_u8, re_u8):
            feats.extend(_glcm(img))
            feats.extend(_lbp(img, 1, 8))
            feats.extend(_lbp(img, 2, 12, 32))
            feats.extend(_gabor(img))

        # Gradient features on NDVI (4)
        gy, gx = np.gradient(ndvi)
        grad = np.sqrt(gx**2 + gy**2)
        feats.extend([grad.mean(), grad.std(), grad.max(), np.percentile(grad, 90)])

        # Spatial uniformity (quadrants + 4x4 blocks) (8)
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

        # Green channel texture
        green_u8 = _normalize_u8(green)
        feats.extend(_glcm(green_u8, distances=(1, 2), angles=(0, np.pi / 2)))
        feats.extend(_lbp(green_u8))
    else:
        feats.extend([0] * 584)  # MS texture-only features

    # ======================== RGB ========================
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
        feats.extend(_glcm(gray, distances=(1, 2), angles=(0, np.pi / 2)))
        feats.extend(_lbp(gray))
        feats.extend(_gabor(gray))
        gy, gx = np.gradient(rgb.mean(axis=2))
        grad = np.sqrt(gx**2 + gy**2)
        feats.extend([grad.mean(), grad.std(), grad.max()])
    else:
        feats.extend([0] * 120)

    # ======================== HS ========================
    pixels = hs.reshape(-1, 125)
    band_means = pixels.mean(axis=0)
    band_stds = pixels.std(axis=0)
    # Use clean bands only (10-110), avoiding noisy sensor edges
    clean_slice = slice(10, 111)
    feats.extend(band_means[clean_slice].tolist())
    feats.extend(band_stds[clean_slice].tolist())

    # HS spectral indices at pixel level
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

    # Spectral heterogeneity (clean bands only)
    cv_pb = band_stds[clean_slice] / (band_means[clean_slice] + EPS)
    feats.extend(
        [
            cv_pb.mean(),
            cv_pb.std(),
            cv_pb[:30].mean(),  # vis: bands 10-40
            cv_pb[30:55].mean(),  # red-edge: bands 40-65
            cv_pb[65:].mean(),  # nir: bands 75-110
        ]
    )

    # Red-edge derivative features
    d1 = np.diff(band_means)
    feats.extend(
        [d1[45:75].max(), float(d1[45:75].argmax()), d1[45:75].mean(), d1[45:75].std()]
    )

    # Region ratios (clean bands: 10-110)
    vis, re_r, nir_r = (
        band_means[10:40].mean(),
        band_means[40:65].mean(),
        band_means[75:110].mean(),
    )
    feats.extend([vis / (nir_r + EPS), re_r / (nir_r + EPS), vis / (re_r + EPS)])

    # HS narrow-band ratios (disease-specific, clean bands)
    pri = (band_means[20] - band_means[30]) / (band_means[20] + band_means[30] + EPS)
    ari = 1.0 / (band_means[25] + EPS) - 1.0 / (band_means[62] + EPS)
    cri = 1.0 / (band_means[18] + EPS) - 1.0 / (band_means[28] + EPS)  # shifted from 15
    feats.extend([pri, ari, cri])

    key_bands = [12, 22, 32, 42, 52, 62, 72, 82, 92, 102]  # clean range 10-110
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

    # ---- Discriminative features from spectral signature analysis ----
    # NIR plateau level (Health~3000 > Rust~2900 > Other~2200)
    nir_plateau = band_means[80:105].mean()
    feats.append(nir_plateau)

    # Green peak prominence (Rust shows bump at 20-30 relative to neighbors)
    green_region = band_means[20:30].mean()
    green_neighbors = (band_means[12:18].mean() + band_means[32:38].mean()) / 2
    green_prominence = green_region - green_neighbors
    feats.extend([green_prominence, green_region / (green_neighbors + EPS)])

    # Red-edge slope steepness (60-75): Health steeper than Other
    re_slope = (band_means[75] - band_means[60]) / 15.0
    feats.append(re_slope)

    # Red-edge inflection point (where max derivative occurs)
    re_derivs = np.diff(band_means[55:80])
    re_inflection = np.argmax(re_derivs) + 55
    feats.append(float(re_inflection))

    # Visible/NIR contrast (Other: high vis, low NIR)
    vis_level = band_means[30:55].mean()
    vis_nir_contrast = vis_level - nir_plateau
    vis_nir_ratio = vis_level / (nir_plateau + EPS)
    feats.extend([vis_nir_contrast, vis_nir_ratio])

    # NIR shape: plateau flatness (Health/Rust flat, Other slopes down)
    nir_early = band_means[80:90].mean()
    nir_late = band_means[95:105].mean()
    nir_slope = nir_late - nir_early
    feats.append(nir_slope)

    # ---- HS zero-pixel fraction (blank/dead tissue indicator) ----
    zero_mask = np.all(hs < 1, axis=2)
    feats.append(zero_mask.sum() / (hs.shape[0] * hs.shape[1]))

    # ---- HS GLCM on critical bands (red-edge B43-59, NIR B80-95) ----
    re_spatial = _normalize_u8(hs[:, :, 43:59].mean(axis=2))
    nir_spatial = _normalize_u8(hs[:, :, 80:95].mean(axis=2))
    for img in (re_spatial, nir_spatial):
        feats.extend(_glcm(img, distances=(1, 2), angles=(0, np.pi / 2)))
        feats.extend(_lbp(img, 1, 8))

    # ---- HS spatial block heterogeneity on critical bands ----
    h_hs, w_hs = hs.shape[0], hs.shape[1]
    for s, e in [(43, 59), (80, 100)]:
        region_map = hs[:, :, s:e].mean(axis=2)
        bh, bw = max(1, h_hs // 4), max(1, w_hs // 4)
        block_means = []
        for bi in range(4):
            for bj in range(4):
                blk = region_map[bi * bh : (bi + 1) * bh, bj * bw : (bj + 1) * bw]
                block_means.append(blk.mean())
        bm = np.array(block_means)
        feats.extend([bm.std(), bm.max() - bm.min(), np.std(bm) / (np.mean(bm) + EPS)])

    # ---- Per-pixel red-edge curvature (2nd derivative) ----
    d2 = np.diff(pixels[:, 43:65], n=2, axis=1)
    d2_max = d2.max(axis=1)
    d2_min = d2.min(axis=1)
    d2_range = d2_max - d2_min
    for arr in (d2_max, d2_min, d2_range):
        feats.extend([arr.mean(), arr.std(), np.median(arr)])

    # ---- HS critical band individual means (43-59, every 2nd) ----
    for b in range(43, 60, 2):
        feats.append(band_means[b])

    # ---- NIR spatial std (diagnostic showed H_as_O has low NIR_std) ----
    nir_region = hs[:, :, 80:100].mean(axis=2)
    feats.extend([nir_region.std(), nir_region.mean() / (nir_region.std() + EPS)])

    # ---- Health vs Rust discriminative: within-sample consistency ----
    # Rust has low CV (consistent), Health has high CV (variable)
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

    # Per-pixel spectral shape consistency
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

    # NIR band consistency (Rust very consistent ~261 std)
    nir_cv = band_stds[80:100] / (band_means[80:100] + EPS)
    feats.extend([nir_cv.mean(), nir_cv.std()])

    # Red-edge consistency
    re_cv = band_stds[50:65] / (band_means[50:65] + EPS)
    feats.extend([re_cv.mean(), re_cv.std()])

    # ======================== Cross-modal ========================
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
