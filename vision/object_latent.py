"""Object-centric latent encoder (option a): image -> one latent per object, no explicit coordinates.

Built on ObjectEncoder (ResNet18 + stride-8 heatmaps). Per object k:
  pooled_k = features averaged under object k's heatmap (attention pooling, local to the object)
  z_k      = MLP([pooled_k, u_k / W, v_k / H])            (D-dim, the only output the decision core sees)
Heads that only exist for training:
  heatmap + soft-argmax (u, v)  : keeps the pooling on the object, not on the arm
  pos_head(z_k) -> (x, y, z)    : forces z_k to carry the object position (mm-level)
  vis_head(z_k) -> visible?     : forces z_k to carry whether the object is seen
  InfoNCE across paired renders : same scene with a different arm (colour, posture) -> same z_k
No global (whole-image) feature enters z_k, so the arm can only leak in through the pooled region.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from vision_encoder import ObjectEncoder


class ObjectLatentEncoder(ObjectEncoder):
    def __init__(self, n_obj=2, norm="gn", pretrained=True, feat=128, z_dim=64,
                 pos_center=(0.45, 0.0, 0.0), pos_scale=(0.15, 0.2, 0.05), **backbone_kw):
        super().__init__(n_obj=n_obj, norm=norm, pretrained=pretrained, feat=feat, **backbone_kw)
        self.z_dim = z_dim
        self.obj_embed = nn.Parameter(torch.zeros(n_obj, feat))       # which object (cube / tape)
        self.to_z = nn.Sequential(nn.Linear(feat + 2, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(),
                                  nn.Linear(256, z_dim))
        self.pos_head = nn.Sequential(nn.Linear(z_dim, 128), nn.GELU(), nn.Linear(128, 3))
        self.vis_head = nn.Sequential(nn.Linear(z_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.register_buffer("pos_center", torch.tensor(pos_center, dtype=torch.float32))
        self.register_buffer("pos_scale", torch.tensor(pos_scale, dtype=torch.float32))

    def _heads(self, f, c3, device):
        out = super()._heads(f, c3, device)
        prob = out["prob"].flatten(2)                                         # (B, K, h*w)
        pooled = prob @ f.flatten(2).transpose(1, 2) + self.obj_embed         # (B, K, feat)
        H, W = f.shape[-2] * self.STRIDE, f.shape[-1] * self.STRIDE
        uvn = out["uv"] / torch.tensor([W, H], device=device, dtype=torch.float32)
        z = self.to_z(torch.cat([pooled, uvn], -1))                           # (B, K, D)
        out["z"] = z
        out["pos"] = self.pos_head(z) * self.pos_scale + self.pos_center      # world metres
        out["vis_logit"] = self.vis_head(z).squeeze(-1)                       # replaces the base vis head
        return out


def info_nce(za, zb, valid, sigma=0.3):
    """Symmetric InfoNCE between paired latents.

    za, zb: (N, D) latents of the same object in the two renders of N scenes; valid: (N,) bool.
    Similarity is a Gaussian kernel on L2-normalised latents, exp(-|a-b|^2 / 2 sigma^2), so the
    usual temperature is tau = sigma^2 (|a-b|^2 = 2 - 2 cos).
    """
    za, zb = za[valid], zb[valid]
    if len(za) < 2:
        return za.sum() * 0
    za, zb = F.normalize(za, dim=-1), F.normalize(zb, dim=-1)
    logits = za @ zb.t() / (sigma ** 2)
    tgt = torch.arange(len(za), device=za.device)
    return 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))


def load_any_encoder(path, device="cpu"):
    """Image encoder from any of our checkpoints -> (encoder, in_hw, has_z).
      train_pp_vision_e2e.py : keys encoder / encoder_config        (ObjectLatentEncoder)
      train_object_latent.py : keys model / z_dim / backbone_kw     (ObjectLatentEncoder)
      train_vision_encoder.py: keys model / roi / in_hw / norm      (ObjectEncoder: keypoints only, no z)
    """
    from vision_encoder import ObjectEncoder
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "encoder" in ck:
        ec = dict(ck["encoder_config"]); in_hw = tuple(ec.pop("in_hw")); ec.pop("goal_frame", None)
        enc = ObjectLatentEncoder(n_obj=2, pretrained=False, z_dim=ec.pop("z_dim"), **ec)
        enc.load_state_dict(ck["encoder"]); has_z = True
    elif "z_dim" in ck:
        enc = ObjectLatentEncoder(n_obj=len(ck["objs"]), norm=ck["norm"], pretrained=False, z_dim=ck["z_dim"],
                                  pos_center=ck["pos_center"], pos_scale=ck["pos_scale"],
                                  **(ck.get("backbone_kw") or {}))
        enc.load_state_dict(ck["model"]); in_hw = tuple(ck["in_hw"]); has_z = True
    else:
        enc = ObjectEncoder(n_obj=len(ck.get("objs", ("cube", "tape"))), norm=ck["norm"], pretrained=False,
                            **(ck.get("backbone_kw") or {}))
        enc.load_state_dict(ck["model"]); in_hw = tuple(ck["in_hw"]); has_z = False
    return enc.to(device).eval(), in_hw, has_z