"""Stage-1 sampling helpers for PlayVox.

``build_cond`` turns sampled frames + object/hand masks into the multiview
conditioning tensor and the raw DINO context. ``joint_sample`` co-denoises the
sparse-structure shape latent and the per-view amodal mask, threading mask
tokens through the DiT each Euler step (flow-matching convention: t 0->1,
x += dt*v). CFG is applied to the shape latent only; the mask gets the
positive pass only. ``decode_shape`` turns the sampled latent into 64^3
occupancy coordinates.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch

MASK_RES = 64
GRID = 64
LORA_TARGETS = ["to_q.shape", "to_kv.shape", "to_out.shape", "to_qkv.shape",
                "shape.to_q", "shape.to_kv", "shape.to_out", "shape.to_qkv",
                "shape.mlp.0", "shape.mlp.2", "adaLN_modulation.1"]




def detect_arch(ckpt_path):
    """Detect arch: prefer the `forehoi_arch`
    stamp; else infer from weights (pointmap_gate / ref_src_emb ctx 4096; else ctx 3072)."""
    blob = torch.load(ckpt_path, map_location="cpu")
    arch = blob.get("forehoi_arch") if isinstance(blob, dict) else None
    sd = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
    if arch and "cond_ctx" in arch:
        return {"use_depth": int(arch["cond_ctx"]) >= 4096, "cond_ctx": int(arch["cond_ctx"])}
    has_pm_gate = any(k.startswith("models.pointmap_gate") for k in sd)
    ref = sd.get("models.ref_src_emb.weight")
    ctx = int(ref.shape[-1]) if ref is not None else (4096 if has_pm_gate else 3072)
    use_depth = has_pm_gate or ctx >= 4096
    return {"use_depth": use_depth, "cond_ctx": ctx}




# ---------------------------------------------------------------------------
# standalone helpers (work with any pipeline that has the right .models)
# ---------------------------------------------------------------------------
def crop_affine_from_mask(masks, padding_factor=1.1):
    """masks (M,1,H,W) -> affine A (M,2,3) for a square bbox crop."""
    M, _, H, W = masks.shape
    device = masks.device
    mask_bool = masks[:, 0] > 0.5
    inf = torch.tensor(1e5, device=device)
    ys = torch.arange(H, device=device).view(1, H, 1).expand(M, H, W).float()
    xs = torch.arange(W, device=device).view(1, 1, W).expand(M, H, W).float()
    y0 = torch.where(mask_bool, ys, inf).flatten(1).min(1)[0].clamp(max=H - 1)
    x0 = torch.where(mask_bool, xs, inf).flatten(1).min(1)[0].clamp(max=W - 1)
    y1 = torch.where(mask_bool, ys, -inf).flatten(1).max(1)[0].clamp(min=0)
    x1 = torch.where(mask_bool, xs, -inf).flatten(1).max(1)[0].clamp(min=0)
    no_fg = (mask_bool.flatten(1).sum(1) == 0)
    y0 = torch.where(no_fg, torch.zeros_like(y0), y0)
    x0 = torch.where(no_fg, torch.zeros_like(x0), x0)
    y1 = torch.where(no_fg, torch.full_like(y1, H - 1), y1)
    x1 = torch.where(no_fg, torch.full_like(x1, W - 1), x1)
    cy = (y0 + y1) * 0.5; cx = (x0 + x1) * 0.5
    side = torch.max(y1 - y0, x1 - x0) * padding_factor
    y0n = (cy - side / 2).clamp(0, H - 1); y1n = (cy + side / 2).clamp(0, H - 1)
    x0n = (cx - side / 2).clamp(0, W - 1); x1n = (cx + side / 2).clamp(0, W - 1)
    scale_y = (y1n - y0n) / (H - 1); scale_x = (x1n - x0n) / (W - 1)
    trans_y = (y0n + y1n - (H - 1)) / (H - 1); trans_x = (x0n + x1n - (W - 1)) / (W - 1)
    A = torch.zeros(M, 2, 3, device=device, dtype=torch.float32)
    A[:, 0, 0] = scale_x; A[:, 1, 1] = scale_y
    A[:, 0, 2] = trans_x; A[:, 1, 2] = trans_y
    return A


def grid_crop(x, A, resolution, mode="bilinear"):
    import torch.nn.functional as F
    B, C = x.shape[0], x.shape[1]
    grid = F.affine_grid(A, [B, C, resolution, resolution], align_corners=False)
    return F.grid_sample(x, grid.to(x.dtype), mode=mode, align_corners=False, padding_mode="zeros")


@torch.no_grad()
def encode_dino(dino, img):
    """img (M,C,518,518) in [0,1] -> (M,1369,1024) patch tokens (drop cls)."""
    return dino(img)[:, 1:]


def zero_pose_latents(backbone, b, device, dtype):
    """Inert zero latents for the kept (but unused) pose modalities."""
    out = {}
    for name, latent in backbone.latent_mapping.items():
        if name == "shape":
            continue
        out[name] = torch.zeros(b, latent.pos_emb.shape[0], latent.input_layer.in_features,
                                device=device, dtype=dtype)
    return out


# ---------------------------------------------------------------------------
# depth/pointmap helpers (standalone — work with TrainingMVSSPipeline AND the
# full Inference pipeline; both expose `.depth_model`). MoGe is the wild source.
# ---------------------------------------------------------------------------
_SSI = None


def _get_ssi():
    global _SSI
    if _SSI is None:
        from forehoi.data.dataset.tdfy.img_and_mask_transforms import ObjectCentricSSI
        _SSI = ObjectCentricSSI(use_scene_scale=True, allow_scale_and_shift_override=True, scale_factor=1.0)
    return _SSI


@torch.no_grad()
def moge_pointmap(depth_model, rgb, dtype=torch.float16):
    """rgb (M,3,H,W) in [0,1] -> (M,3,H,W) PyTorch3D-frame MoGe pointmap (full frame)."""
    from forehoi.pipeline.inference_pipeline_pointmap import camera_to_pytorch3d_camera
    from pytorch3d.transforms import Transform3d
    with torch.autocast("cuda", dtype=dtype):
        out = depth_model(rgb)
    pm = out["pointmaps"].float()                       # (M,H,W,3) MoGe native
    M, H, W, _ = pm.shape
    R = camera_to_pytorch3d_camera(device=pm.device).rotation
    tf = Transform3d().rotate(R).to(pm.device)
    return tf.transform_points(pm.reshape(M, -1, 3)).reshape(M, H, W, 3).permute(0, 3, 1, 2)


@torch.no_grad()
def ssi_normalize_pm(pm, objmask):
    """Per-view ObjectCentricSSI; empty-mask views -> all-NaN (avoids raw .max() crash)."""
    ssi = _get_ssi()
    outs = []
    for i in range(pm.shape[0]):
        if (objmask[i] > 0.5).sum() < 16:        # empty / near-degenerate -> invalid
            outs.append(torch.full_like(pm[i], float("nan"))); continue
        try:
            outs.append(ssi.normalize(pm[i], objmask[i]).pointmap)
        except Exception:
            outs.append(torch.full_like(pm[i], float("nan")))
    return torch.stack(outs, 0)


# ---------------------------------------------------------------------------
# conditioning + joint sampling
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_cond(pipeline, image, obj, hand, device="cuda"):
    """image (n,3,H,W) [0,1]; obj,hand (n,1,H,W) {0,1}. -> cond (1,4096,1024), raw_ctx (1,n,1369,1024)."""
    n = image.shape[0]
    rgb = image.to(device).float()
    obj = obj.to(device).float()
    hand = hand.to(device).float()
    union = ((obj + hand) > 0.5).float()
    with torch.autocast("cuda", dtype=pipeline.dtype):
        A = crop_affine_from_mask(union, padding_factor=1.1)
        rgb_c = grid_crop(rgb, A, 518, "bilinear")
        obj_c = grid_crop(obj, A, 518, "nearest")
        union_c = grid_crop(union, A, 518, "nearest")
        image_img = rgb_c * obj_c
        raw_img = rgb_c * union_c
        mask_img = obj_c
        img_dino = pipeline.models["ss_condition_embedder"].module_list[0]
        mask_dino = pipeline.models["ss_condition_embedder"].module_list[1]
        image_feat = encode_dino(img_dino, image_img)
        mask_feat = encode_dino(mask_dino, mask_img)
        raw_feat = encode_dino(img_dino, raw_img)
        raw_gate = pipeline.models["raw_gate"].weight
        if getattr(pipeline, "use_depth", False):
            # 4th pointmap stream from a MoGe pointmap on the (full-frame) wild image
            # — the `moge` training mode. Object-masked + per-view ObjectCentricSSI + same crop.
            pm = moge_pointmap(pipeline.depth_model, rgb, pipeline.dtype)                        # (n,3,H,W) p3d
            objm = obj > 0.5
            pm = torch.where(objm, pm, torch.full_like(pm, float("nan")))
            pm = ssi_normalize_pm(pm, obj)
            pm_c = grid_crop(pm, A, 518, "nearest")
            pm_c = torch.where(obj_c > 0.5, pm_c, torch.full_like(pm_c, float("nan")))
            emb = pipeline.models["ss_condition_embedder"].module_list[2](pm_c)
            pm_feat = pipeline.models["ss_condition_embedder"].projection_nets[2](emb)
            pm_feat = pm_feat * pipeline.models["pointmap_gate"].weight
            ctx = torch.cat([image_feat, mask_feat, raw_feat * raw_gate, pm_feat], dim=-1)
            ctx = ctx.reshape(1, n, -1, pipeline.cond_ctx_channels)
        else:
            ctx = torch.cat([image_feat, mask_feat, raw_feat * raw_gate], dim=-1).reshape(1, n, -1, 3072)
        ref_src = pipeline.models["ref_src_emb"].weight
        ref_cond = ctx[:, :1] + ref_src[0:1, None, None]
        if n == 1:
            condition = ref_cond
        else:
            src_cond = ctx[:, 1:] + ref_src[1:, None, None]
            condition = torch.cat([ref_cond, src_cond], dim=1)
        cond = pipeline.models["multiview_cond"](condition)
    raw_ctx = raw_feat.reshape(1, n, -1, 1024)
    return cond, raw_ctx




@torch.no_grad()
def joint_sample(pipeline, cond, raw_ctx, n_use, steps=25, cfg=7.0,
                 rescale_t=3.0, cfg_t_max=0.5, device="cuda", seed=None):
    """Joint shape+mask Euler sampling. -> (z_shape (1,4096,8), z_mask {-1,+1} (1,n,1,64,64))."""
    if seed is not None:
        torch.manual_seed(seed)
    gen = pipeline.models["ss_generator"]
    backbone = gen.reverse_fn.backbone
    time_scale = gen.time_scale
    mask_branch = pipeline.models["mask_branch"]
    lm = backbone.latent_mapping
    n_tok = lm["shape"].pos_emb.shape[0]
    in_ch = lm["shape"].input_layer.in_features
    z_shape = torch.randn(1, n_tok, in_ch, device=device)
    z_mask = torch.randn(1, n_use, 1, MASK_RES, MASK_RES, device=device)
    zero_pose = zero_pose_latents(backbone, 1, device, z_shape.dtype)
    t_seq = np.linspace(0, 1, steps + 1)
    t_seq = t_seq / (1 + (rescale_t - 1) * (1 - t_seq))
    for i in range(steps):
        t0 = float(t_seq[i]); dt = float(t_seq[i + 1] - t_seq[i])
        tt = torch.tensor([t0 * time_scale], device=device, dtype=torch.float32)
        with torch.autocast("cuda", dtype=pipeline.dtype):
            mtok = mask_branch.to_tokens(z_mask, raw_ctx)
            latents = {"shape": z_shape, **zero_pose}
            out_pos = backbone(latents, tt, cond, mask_tokens=mtok)
            v_shape = out_pos["shape"].float()
            v_mask = mask_branch.from_tokens(out_pos["mask_out"], n_use).float()
            if cfg and cfg != 1.0 and t0 <= cfg_t_max:
                out_neg = backbone(latents, tt, cond, mask_tokens=mtok, cfg=True)
                v_shape = (1.0 + cfg) * v_shape - cfg * out_neg["shape"].float()
        z_shape = z_shape + dt * v_shape
        z_mask = z_mask + dt * v_mask
    return z_shape, z_mask


@torch.no_grad()
def decode_shape(pipeline, z_shape):
    """z_shape (1,4096,8) -> occupied voxel coords (N,3) int."""
    ss_decoder = pipeline.models["ss_decoder"]
    with torch.autocast("cuda", dtype=pipeline.dtype):
        ss = ss_decoder(z_shape.permute(0, 2, 1).contiguous().view(z_shape.shape[0], 8, 16, 16, 16))
    coords = torch.argwhere(ss > 0)[:, [0, 2, 3, 4]].int()
    return coords[:, 1:].cpu().numpy().astype(np.int32)


# ---------------------------------------------------------------------------
# metrics + IO utils
# ---------------------------------------------------------------------------






