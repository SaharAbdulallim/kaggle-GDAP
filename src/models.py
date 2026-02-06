import timm
import torch
import torch.nn as nn


class SpectralAttention(nn.Module):
    """Channel attention mechanism for spectral data.
    
    Learns importance weights for each spectral channel/band via global pooling
    and small FC network. Useful for emphasizing disease-relevant wavelengths.
    """
    def __init__(self, channels, reduction=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 1)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 1), channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MultiModalClassifier(nn.Module):
    """Multimodal classifier for RGB + Multispectral + Hyperspectral data.
    
    Architecture:
    - RGB: Frozen pretrained backbone + trainable head
    - MS: SpectralAttention → ResNet18 from scratch
    - HS: SpectralAttention → ResNet18 from scratch (on PCA-reduced data)
    - Fusion: Concat or attention-based
    - Classifier: Dropout + Linear layers
    """
    def __init__(self, cfg, hs_channels, num_classes=3):
        super().__init__()
        self.cfg = cfg
        self.use_rgb = cfg.USE_RGB
        self.use_ms = cfg.USE_MS
        self.use_hs = cfg.USE_HS
        
        feat_dims = []
        
        if self.use_rgb:
            self.rgb_encoder = timm.create_model(
                cfg.RGB_BACKBONE,
                pretrained=cfg.RGB_PRETRAINED,
                in_chans=3,
                num_classes=0
            )
            
            if cfg.RGB_FREEZE_ENCODER:
                for param in self.rgb_encoder.parameters():
                    param.requires_grad = False
            
            rgb_feat_dim = self.rgb_encoder.num_features
            self.rgb_head = nn.Sequential(
                nn.Linear(rgb_feat_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.DROPOUT)
            )
            feat_dims.append(256)
        
        if self.use_ms:
            ms_ch = 5 + cfg.MS_ADD_NDVI + cfg.MS_ADD_NDRE
            self.ms_attention = SpectralAttention(ms_ch)
            self.ms_encoder = timm.create_model(
                cfg.MS_BACKBONE,
                pretrained=False,
                in_chans=ms_ch,
                num_classes=0
            )
            feat_dims.append(self.ms_encoder.num_features)
        
        if self.use_hs:
            self.hs_attention = SpectralAttention(hs_channels)
            self.hs_encoder = timm.create_model(
                cfg.HS_BACKBONE,
                pretrained=False,
                in_chans=hs_channels,
                num_classes=0
            )
            feat_dims.append(self.hs_encoder.num_features)
        
        total_feat_dim = sum(feat_dims)
        classifier_in = total_feat_dim
        
        self.classifier = nn.Sequential(
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(classifier_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.DROPOUT * 0.6),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, modalities):
        features = []
        
        if self.use_rgb and 'rgb' in modalities:
            if self.cfg.RGB_FREEZE_ENCODER:
                with torch.no_grad():
                    rgb_feat = self.rgb_encoder(modalities['rgb'])
            else:
                rgb_feat = self.rgb_encoder(modalities['rgb'])
            rgb_feat = self.rgb_head(rgb_feat)
            features.append(rgb_feat)
        
        if self.use_ms and 'ms' in modalities:
            ms = self.ms_attention(modalities['ms'])
            ms_feat = self.ms_encoder(ms)
            features.append(ms_feat)
        
        if self.use_hs and 'hs' in modalities:
            hs = self.hs_attention(modalities['hs'])
            hs_feat = self.hs_encoder(hs)
            features.append(hs_feat)
        
        fused = torch.cat(features, dim=1)
        return self.classifier(fused)
    
    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_total_params(self):
        return sum(p.numel() for p in self.parameters())
