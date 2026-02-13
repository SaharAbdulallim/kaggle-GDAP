# Wheat Disease Classification - Multimodal Remote Sensing

Competition: [Beyond Visible Spectrum AI for Agriculture 2026](https://www.kaggle.com/competitions/beyond-visible-spectrum-ai-for-agriculture-2026)

## Problem

Classify wheat into 3 classes (Healthy/Rust/Other) using multimodal remote sensing:

- RGB: 3 channels (visible)
- Multispectral (MS): 5 channels (visible + NIR)
- Hyperspectral (HS): 101 channels (400-1000nm) ( After filtering noise channels , 10 first 14 last)

## Dataset

This dataset contains multimodal UAV imagery for the classification of wheat diseases. The data was collected to identify the spread of downy mildew and rust at critical growth stages. The dataset has been pre-processed into three modalities: RGB, Multispectral (MS), and Hyperspectral (HS), allowing for the development of multimodal deep learning models.

### Data Acquisition

    Dates: May 3, 2019 (Pre-grouting stage) and May 8, 2019 (Middle grouting stage).
    Equipment: DJI M600 Pro UAV with an S185 snapshot hyperspectral sensor.
    Flight Altitude: 60 meters (Spatial resolution ~4cm/pixel).
    Spectral Range: 450-950nm (Visible to Near-Infrared).
    Spectral Resolution: 4nm.

### Data Modalities

For each sample, three aligned data types are provided:

#### RGB Images (/RGB)

     Format: .png
     Description: True-color images generated from the hyperspectral bands (Red: ~650nm, Green: ~550nm, Blue: ~480nm).

#### Multispectral Data (/MS)

     Format: .tif (GeoTIFF)
     Bands: 5 bands critical for vegetation health analysis:
          Blue (~480nm)
          Green (~550nm)
          Red (~650nm)
          Red Edge (740nm)
          NIR (833nm)

#### Hyperspectral Data (/HS)

     Format: .tif (GeoTIFF)
     Bands: 125 bands (450-950nm).
     Note: While the raw data contains 125 bands, the spectral ends (first ~10 and last ~14 bands) may contain sensor noise.

## Implementation Approach
