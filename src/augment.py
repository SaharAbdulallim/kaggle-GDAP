import numpy as np


def spatial_augment(sample, prob=0.5):
    if np.random.rand() > prob:
        return sample

    aug = sample.copy()
    hs, ms, rgb = aug.get("hs"), aug.get("ms"), aug.get("rgb")

    flip_h = np.random.rand() < 0.5
    flip_v = np.random.rand() < 0.5
    rot_k = np.random.randint(0, 4)

    if hs is not None:
        if flip_h:
            hs = np.flip(hs, axis=1)
        if flip_v:
            hs = np.flip(hs, axis=0)
        if rot_k > 0:
            hs = np.rot90(hs, k=rot_k, axes=(0, 1))
        aug["hs"] = hs.copy()

    if ms is not None:
        if flip_h:
            ms = np.flip(ms, axis=1)
        if flip_v:
            ms = np.flip(ms, axis=0)
        if rot_k > 0:
            ms = np.rot90(ms, k=rot_k, axes=(0, 1))
        aug["ms"] = ms.copy()

    if rgb is not None:
        if flip_h:
            rgb = np.flip(rgb, axis=1)
        if flip_v:
            rgb = np.flip(rgb, axis=0)
        if rot_k > 0:
            rgb = np.rot90(rgb, k=rot_k, axes=(0, 1))
        aug["rgb"] = rgb.copy()

    return aug


def spectral_augment(sample, noise_scale=0.02, scale_range=(0.95, 1.05), prob=0.5):
    if np.random.rand() > prob:
        return sample

    aug = sample.copy()
    hs, ms = aug.get("hs"), aug.get("ms")

    if hs is not None:
        hs = hs.copy()
        if np.random.rand() < 0.5:
            noise = np.random.randn(*hs.shape) * noise_scale * hs.std()
            hs = hs + noise
        if np.random.rand() < 0.5:
            scale = np.random.uniform(*scale_range)
            hs = hs * scale
        if np.random.rand() < 0.3:
            band_scale = np.random.uniform(0.97, 1.03, size=hs.shape[2])
            hs = hs * band_scale[None, None, :]
        aug["hs"] = np.clip(hs, 0, None)

    if ms is not None:
        ms = ms.copy()
        if np.random.rand() < 0.5:
            noise = np.random.randn(*ms.shape) * noise_scale * ms.std()
            ms = ms + noise
        if np.random.rand() < 0.5:
            scale = np.random.uniform(*scale_range)
            ms = ms * scale
        aug["ms"] = np.clip(ms, 0, None)

    return aug


def mixup(sample1, sample2, alpha=0.4):
    if alpha <= 0:
        return sample1

    lam = np.random.beta(alpha, alpha)
    mixed = {"name": f"{sample1['name']}_mix_{sample2['name']}"}

    if "hs" in sample1 and "hs" in sample2:
        mixed["hs"] = lam * sample1["hs"] + (1 - lam) * sample2["hs"]
    elif "hs" in sample1:
        mixed["hs"] = sample1["hs"]

    if (
        "ms" in sample1
        and "ms" in sample2
        and sample1["ms"] is not None
        and sample2["ms"] is not None
    ):
        mixed["ms"] = lam * sample1["ms"] + (1 - lam) * sample2["ms"]
    else:
        mixed["ms"] = sample1.get("ms")

    if (
        "rgb" in sample1
        and "rgb" in sample2
        and sample1["rgb"] is not None
        and sample2["rgb"] is not None
    ):
        mixed["rgb"] = lam * sample1["rgb"] + (1 - lam) * sample2["rgb"]
    else:
        mixed["rgb"] = sample1.get("rgb")

    if "label" in sample1 and "label" in sample2:
        mixed["label"] = sample1["label"]
        mixed["label_mix"] = (lam, sample1["label"], sample2["label"])
    elif "label" in sample1:
        mixed["label"] = sample1["label"]

    if "cls" in sample1:
        mixed["cls"] = sample1["cls"]

    return mixed


def augment_sample(
    sample, spatial_prob=0.7, spectral_prob=0.5, mixup_sample=None, mixup_prob=0.3
):
    aug = spatial_augment(sample, prob=spatial_prob)
    aug = spectral_augment(aug, prob=spectral_prob)

    if mixup_sample is not None and np.random.rand() < mixup_prob:
        aug = mixup(aug, mixup_sample)

    return aug


def create_augmented_batch(
    samples, labels, aug_factor=2, spatial_prob=0.7, spectral_prob=0.5, mixup_prob=0.2
):
    augmented_samples = []
    augmented_labels = []

    for _ in range(aug_factor):
        for i, sample in enumerate(samples):
            mixup_sample = None
            if np.random.rand() < mixup_prob:
                j = np.random.randint(0, len(samples))
                if j != i:
                    mixup_sample = samples[j]

            aug_sample = augment_sample(
                sample, spatial_prob, spectral_prob, mixup_sample, mixup_prob
            )
            augmented_samples.append(aug_sample)
            augmented_labels.append(labels[i])

    return augmented_samples, np.array(augmented_labels)
