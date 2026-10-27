"""Object-position encoder: RGB image -> per-object pixel (u, v) + visibility.

ResNet18 (ImageNet) up to layer3 (stride 16), fused with layer2 (stride 8) so the
heatmaps sit on a stride-8 grid, one heatmap per object, soft-argmax (spatial
softmax expectation, Finn et al. 2016 / Sun et al. 2018) -> sub-pixel (u, v).
Visibility is read from features pooled under each object's heatmap.

uv_to_world() turns (u, v) into (x, y, z) with the calibrated camera and the
known object centre height, so nothing about the arm enters the output.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def bn_to_gn(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            c = child.num_features
            gn = nn.GroupNorm(max(1, c // 16), c)
            gn.weight.data.copy_(child.weight.data)
            gn.bias.data.copy_(child.bias.data)
            setattr(module, name, gn)
        else:
            bn_to_gn(child)
    return module


class ObjectEncoder(nn.Module):
    STRIDE = 8

    def __init__(self, n_obj=2, norm="gn", pretrained=True, feat=128):
        super().__init__()
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet18(weights=w)
        if norm == "gn":
            bn_to_gn(r)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1, self.layer2, self.layer3 = r.layer1, r.layer2, r.layer3

        def gn(c):
            return nn.GroupNorm(max(1, c // 16), c)
        self.top = nn.Sequential(nn.Conv2d(256, feat, 3, padding=1), gn(feat), nn.ReLU(inplace=True))
        self.lat = nn.Sequential(nn.Conv2d(128, feat, 1), gn(feat))
        self.fuse = nn.Sequential(nn.Conv2d(feat, feat, 3, padding=1), gn(feat), nn.ReLU(inplace=True),
                                  nn.Conv2d(feat, feat, 3, padding=1), gn(feat), nn.ReLU(inplace=True))
        self.hm = nn.Conv2d(feat, n_obj, 1)
        self.log_beta = nn.Parameter(torch.zeros(n_obj))   # softmax sharpness, learned
        self.vis = nn.Sequential(nn.Linear(feat + 256, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))
        self.n_obj = n_obj
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, img):
        """img: (B, 3, H, W) float in [0, 1], H and W divisible by 16."""
        x = (img - self.mean) / self.std
        x = self.stem(x)
        c2 = self.layer2(self.layer1(x))
        c3 = self.layer3(c2)
        f = F.interpolate(self.top(c3), size=c2.shape[-2:], mode="bilinear", align_corners=False) + self.lat(c2)
        f = self.fuse(f)
        logits = self.hm(f).float() * self.log_beta.exp().view(1, -1, 1, 1)   # (B, K, h, w)
        B, K, h, w = logits.shape
        prob = logits.flatten(2).softmax(-1)                                  # (B, K, h*w)
        ys = (torch.arange(h, device=img.device, dtype=torch.float32) + 0.5) * self.STRIDE
        xs = (torch.arange(w, device=img.device, dtype=torch.float32) + 0.5) * self.STRIDE
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([gx.flatten(), gy.flatten()], -1)                  # (h*w, 2) input pixels
        uv = prob @ grid                                                      # (B, K, 2)
        pooled = prob @ f.float().flatten(2).transpose(1, 2)                  # (B, K, feat)
        glob = c3.float().mean((2, 3)).unsqueeze(1).expand(-1, K, -1)
        vis_logit = self.vis(torch.cat([pooled, glob], -1)).squeeze(-1)       # (B, K)
        return {"logits": logits, "prob": prob.view(B, K, h, w), "uv": uv, "vis_logit": vis_logit}


def heatmap_target(uv, h, w, stride, sigma_cells=1.5):
    """Gaussian target distribution on the heatmap grid. uv: (B, K, 2) input pixels."""
    ys = (torch.arange(h, device=uv.device) + 0.5) * stride
    xs = (torch.arange(w, device=uv.device) + 0.5) * stride
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    d2 = (gx[None, None] - uv[..., 0, None, None]) ** 2 + (gy[None, None] - uv[..., 1, None, None]) ** 2
    t = torch.exp(-d2 / (2 * (sigma_cells * stride) ** 2))
    return t / t.flatten(2).sum(-1).clamp_min(1e-12)[..., None, None]


def uv_to_world(uv, K, cam_pos, cam_xmat, z_plane):
    """Back-project pixels onto the horizontal plane z = z_plane (object centre height).

    uv (N, 2) full-image continuous pixels; K (N, 3, 3) or (3, 3); cam_pos (N, 3); cam_xmat (N, 9);
    z_plane scalar or (N,). Returns (N, 3) world points. MuJoCo camera: looks along -z, y up.
    """
    uv = np.atleast_2d(uv)
    N = len(uv)
    K = np.broadcast_to(K, (N, 3, 3))
    R = np.asarray(cam_xmat).reshape(N, 3, 3)
    dc = np.stack([(uv[:, 0] - K[:, 0, 2]) / K[:, 0, 0], -(uv[:, 1] - K[:, 1, 2]) / K[:, 1, 1], -np.ones(N)], -1)
    dw = np.einsum("nij,nj->ni", R, dc)
    t = (np.broadcast_to(z_plane, (N,)) - cam_pos[:, 2]) / dw[:, 2]
    return cam_pos + t[:, None] * dw
