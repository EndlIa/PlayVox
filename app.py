"""PlayVox: interactive ForeHOI stage-1 voxel demo.

Pipeline: video -> SAM2 hand/object masks -> ForeHOI sparse structure + amodal
mask -> 64^3 occupied voxels. No SLAT, mesh, Gaussian, DA3, or FoundationPose.
"""
from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from datetime import datetime
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


_CUBE_FACES = (
    ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
    ((-1, 0, 0), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))),
    ((0, 1, 0), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))),
    ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
    ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
    ((0, 0, -1), ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))),
)


def build_voxel_mesh(coords: np.ndarray):
    """Return (vertices, faces) for the exposed cube faces of occupied voxels.

    vertices: float32 [N, 3] in normalized [-0.5, 0.5] space.
    faces:    int32 [M, 3] triangle indices into vertices.
    """
    coords = np.asarray(coords, dtype=np.int32)
    if coords.size == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)

    occupied = set(map(tuple, coords.tolist()))
    verts, index, faces = [], {}, []

    def vid(point):
        idx = index.get(point)
        if idx is None:
            idx = len(verts)
            index[point] = idx
            verts.append(point)
        return idx

    for x, y, z in occupied:
        for (nx, ny, nz), corners in _CUBE_FACES:
            if (x + nx, y + ny, z + nz) in occupied:
                continue
            a, b, c, d = (vid((x + cx, y + cy, z + cz)) for cx, cy, cz in corners)
            faces.append((a, b, c))
            faces.append((a, c, d))

    vertices = np.asarray(verts, dtype=np.float32) / 64.0 - 0.5
    return vertices, np.asarray(faces, dtype=np.int32)


def voxel_figure(coords: np.ndarray) -> go.Figure:
    """Render occupied voxels as opaque cubes."""
    coords = np.asarray(coords, dtype=np.int32)
    if coords.size == 0:
        return go.Figure().update_layout(title="No occupied voxels")

    vertices, faces = build_voxel_mesh(coords)
    fig = go.Figure(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=vertices[:, 2],
            intensitymode="vertex",
            colorscale="Turbo",
            showscale=False,
            flatshading=True,
            opacity=1.0,
            hoverinfo="skip",
            lighting={
                "ambient": 0.55,
                "diffuse": 0.85,
                "specular": 0.15,
                "roughness": 0.85,
                "fresnel": 0.1,
            },
            lightposition={"x": 120, "y": 200, "z": 160},
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


def export_voxels(coords: np.ndarray, path: Path) -> Path:
    """Write voxels.npz + voxels.obj + meta.json into a single zip."""
    coords = np.asarray(coords, dtype=np.uint8)
    vertices, faces = build_voxel_mesh(coords)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, coords=coords, resolution=64)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("voxels.npz", buffer.getvalue())
        obj = [
            f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices
        ] + [
            f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces
        ]
        zf.writestr("voxels.obj", "\n".join(obj) + "\n")
        zf.writestr(
            "meta.json",
            json.dumps(
                {
                    "format": "playvox-voxels/1",
                    "resolution": 64,
                    "voxel_count": int(len(coords)),
                    "grid": "occupancy indices in [0, 63]",
                    "obj_space": "normalized [-0.5, 0.5]; no metric scale",
                    "views": len(ui.app_state.frames),
                    "steps": STEPS,
                    "cfg": CFG,
                    "seed": SEED,
                    "checkpoint": Path(CKPT).name,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                },
                indent=2,
            ),
        )
    return path


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
        z_shape, _z_mask = C.joint_sample(
            pipeline, cond, raw_ctx, len(ui.app_state.frames),
            steps=STEPS, cfg=CFG, device=DEVICE, seed=seed,
        )
        coords = C.decode_shape(pipeline, z_shape)
    ui.app_state.voxels = coords
    fig = voxel_figure(coords)
    return fig, f"PlayVox complete: {len(coords):,} occupied voxels; {len(ui.app_state.frames)} views; {STEPS} steps."


SEED = 42

with gr.Blocks(title="PlayVox") as demo:
    gr.Image(
        value=str(HERE / "assets" / "playvox.png"),
        show_label=False, container=False, interactive=False,
        height=90, show_download_button=False, show_fullscreen_button=False,
    )
    gr.Markdown("**ForeHOI stage-1 voxel completion** — SAM2 masks → 64³ occupied voxels")
    gr.Markdown("上传视频后标记手的位置和物体的位置, 确认无误后点击生成即可")

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Video", format="mp4", height=420)
            num_samples = gr.Slider(2, 12, value=6, step=1, label="Frames to sample")
        with gr.Column():
            first_frame = gr.Image(
                label="Click object/hand points",
                interactive=False, type="numpy", height=420, visible=False,
            )

    mask_type = gr.Radio(["object", "hand"], value="object", label="Point type")
    with gr.Row():
        clear_obj = gr.Button("Clear object points")
        clear_hand = gr.Button("Clear hand points")
        clear_all = gr.Button("Clear all points")
    point_info = gr.Textbox(label="Point info", interactive=False)

    generate_btn = gr.Button("Generate", variant="primary")
    voxel_output = gr.Plot(label="Voxel reconstruction")
    download_btn = gr.DownloadButton(
        "Download voxels (.zip)", value=None, interactive=False
    )
    status = gr.Textbox(label="Status", interactive=False)

    disabled_download = lambda: gr.update(value=None, interactive=False)

    def on_video(video_path, frames):
        if not video_path:
            return gr.update(), "Upload a video first.", None, "", disabled_download()
        frame, msg, _n, _btn = ui.process_video_upload(video_path, frames)
        return frame, msg, None, "", disabled_download()

    def select_point(mask_type_value, evt: gr.SelectData):
        msg, frame = ui.handle_point_selection(mask_type_value, evt)
        return msg, frame, None, "", disabled_download()

    def clear_points(kind):
        fn = {
            "object": ui.clear_object_points,
            "hand": ui.clear_hand_points,
            "all": ui.clear_all_points,
        }[kind]
        msg, frame = fn()
        return msg, frame, None, "", disabled_download()

    def generate_all():
        ok, msg = ui.generate_masks()
        if not ok:
            return msg, None, disabled_download()
        fig, result = reconstruct_voxels(SEED)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = export_voxels(ui.app_state.voxels, OUTPUT_DIR / f"voxels_{stamp}.zip")
        return f"{msg} {result}", fig, gr.update(value=str(path), interactive=True)

    video_input.upload(
        on_video,
        inputs=[video_input, num_samples],
        outputs=[first_frame, status, voxel_output, point_info, download_btn],
    )
    num_samples.release(
        on_video,
        inputs=[video_input, num_samples],
        outputs=[first_frame, status, voxel_output, point_info, download_btn],
    )
    first_frame.select(
        select_point,
        inputs=[mask_type],
        outputs=[point_info, first_frame, voxel_output, status, download_btn],
    )
    clear_obj.click(lambda: clear_points("object"),
                    outputs=[point_info, first_frame, voxel_output, status, download_btn])
    clear_hand.click(lambda: clear_points("hand"),
                     outputs=[point_info, first_frame, voxel_output, status, download_btn])
    clear_all.click(lambda: clear_points("all"),
                    outputs=[point_info, first_frame, voxel_output, status, download_btn])
    generate_btn.click(
        generate_all,
        outputs=[status, voxel_output, download_btn],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=PORT, debug=False, share=False)
