"""PlayVox: interactive ForeHOI stage-1 voxel demo.

Pipeline: video -> SAM2 hand/object masks -> ForeHOI sparse structure + amodal
mask -> 64^3 occupied voxels. No SLAT, mesh, Gaussian, DA3, or FoundationPose.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Must be set before CUDA initialises: avoids fragmentation on 22-24GB cards.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("PLAYVOX_DATA", str(Path.home() / "playvox" / "assets")))
OUTPUT_DIR = Path(os.environ.get("PLAYVOX_OUTPUT", str(HERE / "output_wild")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", str(OUTPUT_DIR))

sys.path.insert(0, str(HERE))

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import torch
from PIL import Image

import tools  # noqa: F401
import tools.stage1 as C
from tools.pipeline import build_pipeline
import ui

ui.TMP_DIR = str(OUTPUT_DIR)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = str(DATA_ROOT / "checkpoints" / "forehoi.ckpt")
PIPELINE_CONFIG = os.environ.get(
    "PLAYVOX_PIPELINE_CONFIG", str(DATA_ROOT / "checkpoints" / "pipeline.yaml")
)
SAM2_CKPT = str(DATA_ROOT / "weights" / "sam2.1_hiera_large.pt")
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
STEPS = int(os.environ.get("PLAYVOX_STEPS", "10"))
CFG = float(os.environ.get("PLAYVOX_CFG", "7.0"))
PORT = int(os.environ.get("GRADIO_SERVER_PORT", "7871"))

print("[PlayVox] loading SAM2 ...", flush=True)
ui.init_sam2(SAM2_CKPT, SAM2_CFG, DEVICE)
print("[PlayVox] loading stage-1 ...", flush=True)
pipeline = build_pipeline(CKPT, DEVICE, config_path=PIPELINE_CONFIG)
print(f"[PlayVox] ready: steps={STEPS}, cfg={CFG}, device={DEVICE}", flush=True)


def voxel_figure(coords: np.ndarray) -> go.Figure:
    coords = np.asarray(coords, dtype=np.float32)
    if coords.size == 0:
        return go.Figure().update_layout(title="No occupied voxels")
    xyz = (coords + 0.5) / 64.0 - 0.5
    fig = go.Figure(
        go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
            mode="markers",
            marker={"size": 4, "color": xyz[:, 2], "colorscale": "Turbo", "opacity": 0.9},
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"PlayVox occupancy ({len(coords):,} voxels)",
        scene={
            "aspectmode": "cube",
            "xaxis": {"title": "X", "range": [-0.5, 0.5]},
            "yaxis": {"title": "Y", "range": [-0.5, 0.5]},
            "zaxis": {"title": "Z", "range": [-0.5, 0.5]},
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
    )
    return fig


def amodal_mask_image(z_mask: torch.Tensor | None) -> Image.Image | None:
    if z_mask is None:
        return None
    pred = (z_mask[0, :, 0] > 0).detach().cpu().numpy().astype(np.uint8) * 255
    rows = [row for row in pred[:8]]
    if not rows:
        return None
    canvas = rows[0]
    for row in rows[1:]:
        canvas = np.concatenate([canvas, np.full((4, 64), 128, np.uint8), row], axis=0)
    return Image.fromarray(canvas, mode="L").resize((256, 256), Image.Resampling.NEAREST)


def reconstruct_voxels(seed):
    if not ui.app_state.frames:
        raise gr.Error("No frames available. Upload and process a video first.")
    if not ui.app_state.object_masks:
        raise gr.Error("Generate SAM2 masks first.")
    seed = int(seed) if seed is not None else 42
    rgb = torch.stack([
        torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        for frame in ui.app_state.frames
    ])
    obj = torch.stack([
        torch.from_numpy(mask.astype(np.float32))[None]
        for mask in ui.app_state.object_masks
    ])
    hand = torch.stack([
        torch.from_numpy(mask.astype(np.float32))[None]
        for mask in ui.app_state.hand_masks
    ])
    with torch.inference_mode():
        cond, raw_ctx = C.build_cond(pipeline, rgb, obj, hand, DEVICE)
        z_shape, z_mask = C.joint_sample(
            pipeline, cond, raw_ctx, len(ui.app_state.frames),
            steps=STEPS, cfg=CFG, device=DEVICE, seed=seed,
        )
        coords = C.decode_shape(pipeline, z_shape)
    fig = voxel_figure(coords)
    mask_img = amodal_mask_image(z_mask)
    return fig, mask_img, f"PlayVox complete: {len(coords):,} occupied voxels; {len(ui.app_state.frames)} views; {STEPS} steps."


with gr.Blocks(title="PlayVox") as demo:
    gr.Markdown("# PlayVox\n**ForeHOI stage-1 voxel completion** — SAM2 masks → 64³ occupied voxels")
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Video", format="mp4", height=420)
            num_samples = gr.Slider(2, 12, value=6, step=1, label="Frames to sample")
            seed_input = gr.Number(label="Seed", value=42, precision=0)
            process_btn = gr.Button("Process Video", variant="primary")
        with gr.Column():
            first_frame = gr.Image(label="First frame: click object/hand points", interactive=False, type="numpy", height=420, visible=False)
            masked_video = gr.Video(label="SAM2 masks", height=300, visible=False)
            status = gr.Textbox(label="Status", interactive=False)
    mask_type = gr.Radio(["object", "hand"], value="object", label="Point type")
    with gr.Row():
        clear_obj = gr.Button("Clear object points")
        clear_hand = gr.Button("Clear hand points")
        clear_all = gr.Button("Clear all points")
    point_info = gr.Textbox(label="Point info", interactive=False)
    generate_btn = gr.Button("Generate SAM2 masks", variant="primary")
    reconstruct_btn = gr.Button("Generate PlayVox", variant="primary")
    with gr.Row():
        voxel_output = gr.Plot(label="Voxel reconstruction")
        amodal_output = gr.Image(label="Predicted amodal masks")
    result_status = gr.Textbox(label="PlayVox result", interactive=False)

    process_btn.click(
        ui.process_video_upload,
        inputs=[video_input, num_samples],
        outputs=[first_frame, status, num_samples, process_btn],
    )
    first_frame.select(
        ui.handle_point_selection,
        inputs=[mask_type],
        outputs=[point_info, first_frame],
    )
    clear_obj.click(ui.clear_object_points, outputs=[point_info, first_frame])
    clear_hand.click(ui.clear_hand_points, outputs=[point_info, first_frame])
    clear_all.click(ui.clear_all_points, outputs=[point_info, first_frame])
    generate_btn.click(
        ui.generate_masks_and_video,
        outputs=[first_frame, masked_video, status, reconstruct_btn],
    )
    reconstruct_btn.click(
        reconstruct_voxels,
        inputs=[seed_input],
        outputs=[voxel_output, amodal_output, result_status],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=PORT, debug=False, share=False)
