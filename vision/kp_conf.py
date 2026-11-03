"""Keypoint confidence from the encoder's own heatmap -- no labels needed.

conf = probability mass inside the densest (2r+1) x (2r+1) window of the heatmap, in [0, 1]:
  ~1    the heatmap is one tight blob (object seen clearly)
  low   the mass is spread out or split over several places (occluded, confused with something else)
Used as the token's visibility and for the object-permanence filter (keep the last confident keypoint).
"""
import torch
import torch.nn.functional as F


def heatmap_conf(prob, r=2):
    """prob (B, K, h, w), sums to 1 per (B, K) -> (B, K) confidence"""
    k = 2 * r + 1
    win = F.avg_pool2d(prob.float(), k, stride=1, padding=r, count_include_pad=True) * (k * k)
    return win.flatten(2).max(-1).values.clamp(0, 1)