"""
CSRNet Counter - Tile-Based Version (Improved for Stadium Scenes)
"""

import torch
import torch.nn as nn
import torchvision
import cv2
import numpy as np
import time
import os
import config


# =========================================================
# CSRNet Architecture (UNCHANGED)
# =========================================================

class CSRNet(nn.Module):

    def __init__(self, load_vgg_weights=True):
        super(CSRNet, self).__init__()

        self.frontend = self._make_frontend(load_vgg_weights)
        self.backend = self._make_backend()
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self._initialize_weights()

    def _make_frontend(self, load_weights):
        if load_weights:
            vgg16 = torchvision.models.vgg16(
                weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1
            )
        else:
            vgg16 = torchvision.models.vgg16(weights=None)

        return nn.Sequential(*list(vgg16.features.children())[:23])

    def _make_backend(self):
        return nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(True),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2), nn.ReLU(True),
        )

    def _initialize_weights(self):
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.output_layer.weight, std=0.01)
        nn.init.constant_(self.output_layer.bias, 0)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x


# =========================================================
# CSRNet Counter (Tile-Based)
# =========================================================

class CSRNetCounter:

    def __init__(self, weights_path=None, device=None):

        self.device = device if device else config.DEVICE

        print("Building CSRNet...")
        self.model = CSRNet(load_vgg_weights=True)

        if weights_path and os.path.exists(weights_path):
            self._load_weights(weights_path)
        else:
            print("⚠ Using ImageNet-only initialization")

        self.model.to(self.device)
        self.model.eval()

        if config.USE_FP16 and self.device.type == "cuda":
            self.model.half()

        self._warmup()

        print(f"✓ CSRNet ready on {self.device}")

    # =========================================================

    def _load_weights(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]

        self.model.load_state_dict(checkpoint, strict=True)
        print("✓ Pretrained CSRNet weights loaded")

    # =========================================================

    def _warmup(self):
        dummy = torch.zeros(1, 3, 480, 640, device=self.device)
        with torch.no_grad():
            _ = self.model(dummy)
        print("✓ Warm-up complete")

    # =========================================================
    # Preprocessing (UNCHANGED)
    # =========================================================

    def preprocess(self, image):

        orig_h, orig_w = image.shape[:2]

        target_h = (orig_h // 8) * 8
        target_w = (orig_w // 8) * 8

        target_h = max(target_h, 64)
        target_w = max(target_w, 64)

        image = cv2.resize(image, (target_w, target_h))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        image = (image - mean) / std
        image = image.unsqueeze(0).to(self.device)

        if config.USE_FP16 and self.device.type == "cuda":
            image = image.half()

        return image, (orig_h, orig_w), (target_h, target_w)

    # =========================================================
    # SINGLE TILE PREDICTION
    # =========================================================

    def _predict_single(self, image):

        input_tensor, orig_size, resized_size = self.preprocess(image)

        with torch.no_grad():
            density = self.model(input_tensor)

        count = density.sum().item()

        # Area correction
        orig_h, orig_w = orig_size
        resized_h, resized_w = resized_size
        area_ratio = (orig_h * orig_w) / (resized_h * resized_w)

        count *= area_ratio

        density_np = density.squeeze().cpu().float().numpy()

        return count, density_np

    # =========================================================
    # TILE-BASED PREDICTION
    # =========================================================

    def predict(self, image, return_density_map=True):

        start = time.time()

        h, w = image.shape[:2]

        grid = 2
        overlap_ratio = 0.15

        tile_h = h // grid
        tile_w = w // grid

        overlap_h = int(tile_h * overlap_ratio)
        overlap_w = int(tile_w * overlap_ratio)

        total_count = 0.0

        density_full = np.zeros((h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)

        for r in range(grid):
            for c in range(grid):

                y1 = max(r * tile_h - overlap_h, 0)
                y2 = min((r + 1) * tile_h + overlap_h, h)
                x1 = max(c * tile_w - overlap_w, 0)
                x2 = min((c + 1) * tile_w + overlap_w, w)

                tile = image[y1:y2, x1:x2]

                count, density = self._predict_single(tile)

                total_count += count  # Use tile-level count only

                if return_density_map:
                    density_resized = cv2.resize(
                        density,
                        (x2 - x1, y2 - y1),
                        interpolation=cv2.INTER_LINEAR
                    )

                    density_full[y1:y2, x1:x2] += density_resized
                    weight_map[y1:y2, x1:x2] += 1.0

        # Normalize density only for visualization
        if return_density_map:
            weight_map[weight_map == 0] = 1.0
            density_full /= weight_map

        # Overlap compensation for count
        # Since tiles overlap, we reduce by approximate overlap factor
        overlap_correction = 1.0 - overlap_ratio
        total_count *= overlap_correction

        fps = 1.0 / (time.time() - start)

        result = {
            "count": float(total_count),
            "fps": fps
        }

        if return_density_map:
            result["density_map"] = density_full

        return result
    # =========================================================

    def visualize_density(self, image, density_map, alpha=0.4):

        if density_map is None:
            return image

        h, w = image.shape[:2]

        density = cv2.resize(
            density_map.astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_CUBIC
        )

        density -= density.min()
        if density.max() > 0:
            density /= density.max()

        density = (density * 255).astype(np.uint8)
        density = cv2.applyColorMap(density, cv2.COLORMAP_JET)

        return cv2.addWeighted(image, 1 - alpha, density, alpha, 0)