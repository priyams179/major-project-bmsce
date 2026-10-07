"""
UAV Flood Mapping Semantic Segmentation Model (DeepLabV3+ Architecture).
Designed for high-resolution aerial disaster response.

Predicts:
  - 10-Class Multi-Damage Map (Background, Flooded Building, Non-Flooded Building,
    Flooded Road, Non-Flooded Road, Water, Tree, Vehicle, Pool, Grass)
  - Binary Flood Inundation Extent (Inundated vs Dry)

Includes Focal-Dice Combined Loss for dealing with extreme class imbalance.
"""

import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates=(6, 12, 18), out_channels=256):
        super(ASPP, self).__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        ]
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ with ResNet-50 backbone for UAV Aerial Flood Mapping.
    """

    def __init__(self, num_classes=10, in_channels=3, pretrained=True):
        super(DeepLabV3Plus, self).__init__()
        self.num_classes = num_classes

        # ResNet-50 backbone
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        # Handle custom input channels if needed
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        self.initial = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )

        self.layer1 = resnet.layer1  # Low-level features (256 channels)
        self.layer2 = resnet.layer2  # (512 channels)
        self.layer3 = resnet.layer3  # (1024 channels)
        self.layer4 = resnet.layer4  # High-level features (2048 channels)

        # ASPP module
        self.aspp = ASPP(in_channels=2048, atrous_rates=(6, 12, 18), out_channels=256)

        # Low-level projection
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # Decoder fusion
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        x0 = self.initial(x)
        low_level_feat = self.layer1(x0)  # 1/4 resolution, 256 ch
        x2 = self.layer2(low_level_feat)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)              # 1/32 resolution, 2048 ch

        aspp_feat = self.aspp(x4)
        aspp_upsampled = F.interpolate(
            aspp_feat, size=low_level_feat.shape[-2:], mode="bilinear", align_corners=False
        )

        low_proj = self.low_level_proj(low_level_feat)
        fused = torch.cat([aspp_upsampled, low_proj], dim=1)
        out = self.decoder(fused)

        # Upsample back to original image size
        return F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)


# ==========================================
# Loss Functions: Focal + Dice Combined Loss
# ==========================================

class FocalDiceLoss(nn.Module):
    """
    Combined Focal Loss and Multi-Class Dice Loss for handling
    imbalanced aerial flood datasets.
    """

    def __init__(self, num_classes=10, gamma=2.0, alpha=0.5, smooth=1e-5):
        super(FocalDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W]
        targets: [B, H, W]
        """
        # 1. Focal Cross-Entropy Loss
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma * ce_loss).mean()

        # 2. Multi-Class Dice Loss
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets.clamp(0, self.num_classes - 1), num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        dice_losses = []
        for c in range(self.num_classes):
            p_c = probs[:, c, :, :]
            t_c = targets_one_hot[:, c, :, :]
            intersection = torch.sum(p_c * t_c, dim=(1, 2))
            cardinality = torch.sum(p_c + t_c, dim=(1, 2))
            dice_c = 1.0 - (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
            dice_losses.append(dice_c.mean())

        mean_dice = torch.stack(dice_losses).mean()

        return self.alpha * focal_loss + (1.0 - self.alpha) * mean_dice


# ==========================================
# Flood Inference Wrapper
# ==========================================

class UAVFloodSegmentor:
    """
    High-level inferencing wrapper for DeepLabV3+ Flood Segmentor.
    """

    FLOOD_CLASSES = [1, 3, 5]  # Building Flooded, Road Flooded, Water

    def __init__(self, weights_path=None, num_classes=10, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.num_classes = num_classes
        self.model = DeepLabV3Plus(num_classes=num_classes, pretrained=False).to(self.device)

        if weights_path is not None and torch.cuda.is_available():
            print(f"Loading segmentation checkpoint from {weights_path}...")
            state = torch.load(weights_path, map_location=self.device)
            if "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"])
            else:
                self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, image_np, img_size=(512, 512)):
        """
        Runs inference on RGB numpy image (H, W, 3).

        Returns:
            pred_mask: 2D numpy array with class IDs (original resolution H, W)
            flood_mask: 2D binary numpy array (1 = Inundated, 0 = Dry)
            prob_map: 2D float array of maximum flood probability
        """
        orig_h, orig_w = image_np.shape[:2]

        # Preprocess
        resized = cv2.resize(image_np, (img_size[1], img_size[0]))
        norm = resized.astype(np.float32) / 255.0
        norm = (norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()

        # Resize predicted mask back to original resolution with nearest neighbor
        pred_mask = cv2.resize(pred.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # Binary flood map: pixel is 1 if class is Water(5), Flooded Road(3), or Flooded Building(1)
        flood_mask = np.isin(pred_mask, self.FLOOD_CLASSES).astype(np.uint8)

        # Flood probability map
        flood_probs = probs[:, self.FLOOD_CLASSES, :, :].sum(dim=1).squeeze(0).cpu().numpy()
        flood_prob_map = cv2.resize(flood_probs, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        return pred_mask, flood_mask, flood_prob_map

    def predict_smoothed_flood(
        self,
        image_np,
        img_size=(512, 512),
        gaussian_ksize=15,
        morph_ksize=7,
        prob_threshold=0.35,
    ):
        """
        Produces regularized, smoothed continuous flood likelihood map and cohesive flood extent,
        eliminating micro-pixel noise on lawns, trees, or shoreline gravel.
        """
        pred_mask, raw_flood_mask, flood_prob_map = self.predict(image_np, img_size=img_size)

        # 1. Spatial Gaussian Smoothing on continuous probability map
        if gaussian_ksize > 1:
            if gaussian_ksize % 2 == 0:
                gaussian_ksize += 1
            smoothed_prob = cv2.GaussianBlur(flood_prob_map, (gaussian_ksize, gaussian_ksize), 0)
        else:
            smoothed_prob = flood_prob_map

        # 2. Threshold smoothed probability to get cohesive water extent
        smoothed_flood_mask = (smoothed_prob >= prob_threshold).astype(np.uint8)

        # 3. Morphological closing to seal internal water holes and smooth ragged boundaries
        if morph_ksize > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
            smoothed_flood_mask = cv2.morphologyEx(smoothed_flood_mask, cv2.MORPH_CLOSE, kernel)
            smoothed_flood_mask = cv2.morphologyEx(smoothed_flood_mask, cv2.MORPH_OPEN, kernel)

        return pred_mask, smoothed_flood_mask, smoothed_prob

