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
    return [
        band.mean(),
        band.std(),
        *[np.percentile(band, p) for p in (5, 10, 25, 50, 75, 90, 95)],
        skew(band.ravel()),
        kurtosis(band.ravel()),
    ]


def _index_stats(idx):
    return [
        idx.mean(),
        idx.std(),
        *[np.percentile(idx, p) for p in (5, 10, 25, 50, 75, 90, 95)],
        skew(idx.ravel()),
        kurtosis(idx.ravel()),
        idx.max() - idx.min(),
    ]


def _array_stats(arr):
    return [
        arr.mean(),
        arr.std(),
        *[np.percentile(arr, p) for p in (10, 25, 50, 75, 90)],
        skew(arr),
        kurtosis(arr),
        arr.max() - arr.min(),
    ]


# ================================================================== MAIN
def extract(sample) -> np.ndarray:
    hs, ms, rgb = sample["hs"], sample["ms"], sample["rgb"]
    feats = []

    # ======================== MS ========================
    if ms is not None:
        nir, red, re, green, blue = (
            ms[:, :, 4],
            ms[:, :, 2],
            ms[:, :, 3],
            ms[:, :, 1],
            ms[:, :, 0],
        )

        # Per-band stats (5 x 11 = 55)
        for b in range(5):
            feats.extend(_band_stats(ms[:, :, b]))

        # Vegetation indices (5 x 12 = 60)
        ndvi = (nir - red) / (nir + red + EPS)
        ndre = (nir - re) / (nir + re + EPS)
        gndvi = (nir - green) / (nir + green + EPS)
        savi = 1.5 * (nir - red) / (nir + red + 0.5 + EPS)
        evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + EPS)
        for idx in (ndvi, ndre, gndvi, savi, evi):
            feats.extend(_index_stats(idx))

        # Band ratios (10)
        for i in range(5):
            for j in range(i + 1, 5):
                feats.append(ms[:, :, i].mean() / (ms[:, :, j].mean() + EPS))

        # Texture: GLCM + LBP + Gabor on NDVI, NIR, RedEdge
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
        feats.extend([0] * 700)

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
    feats.extend(band_means.tolist())
    feats.extend(band_stds.tolist())

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

    # Spectral heterogeneity
    cv_pb = band_stds / (band_means + EPS)
    feats.extend(
        [
            cv_pb.mean(),
            cv_pb.std(),
            cv_pb[:40].mean(),
            cv_pb[40:65].mean(),
            cv_pb[75:].mean(),
        ]
    )

    # Red-edge derivative features
    d1 = np.diff(band_means)
    feats.extend(
        [d1[45:75].max(), float(d1[45:75].argmax()), d1[45:75].mean(), d1[45:75].std()]
    )

    # Region ratios
    vis, re_r, nir_r = (
        band_means[:40].mean(),
        band_means[40:65].mean(),
        band_means[75:].mean(),
    )
    feats.extend([vis / (nir_r + EPS), re_r / (nir_r + EPS), vis / (re_r + EPS)])

    # HS narrow-band ratios (disease-specific)
    pri = (band_means[20] - band_means[30]) / (band_means[20] + band_means[30] + EPS)
    ari = 1.0 / (band_means[25] + EPS) - 1.0 / (band_means[62] + EPS)
    cri = 1.0 / (band_means[15] + EPS) - 1.0 / (band_means[25] + EPS)
    feats.extend([pri, ari, cri])

    key_bands = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
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

    for s, e in [(0, 30), (30, 50), (50, 70), (70, 100), (100, 125)]:
        region = band_means[s:e]
        feats.extend([region.mean(), region.std()])

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

        hs_ranges = [(5, 15), (20, 30), (38, 48), (55, 65), (80, 100)]
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
