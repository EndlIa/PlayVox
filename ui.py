"""SAM2 masking + shared Gradio helpers for PlayVox.

Deliberately independent of the ``forehoi`` package: this module only handles
video frames, SAM2 mask propagation, point picking and the masked-video preview.
``app.py`` owns the model and reads the UI state defined here.
"""

import os
import tempfile
import subprocess

import cv2
import numpy as np
import torch
import gradio as gr

try:
    from gradio_litmodel3d import LitModel3D
except ImportError:
    print("Warning: gradio_litmodel3d not found. Falling back to gr.Model3D.")
    LitModel3D = None

# ── module config (apps may override) ────────────────────────────────────────
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_wild")
MAX_SEED = np.iinfo(np.int32).max
OBJECT_OBJ_ID = 0
HAND_OBJ_ID = 1
_SAM2 = None


def init_sam2(ckpt, cfg, device="cuda"):
    """Load the SAM2 video predictor once (each app process calls this)."""
    global _SAM2
    from sam2.build_sam import build_sam2_video_predictor
    _SAM2 = build_sam2_video_predictor(cfg, ckpt, device=device)
    return _SAM2


# ── video frame sampling ──────────────────────────────────────────────────────
def convert_video_to_mp4(input_path, output_path):
    subprocess.run(['ffmpeg', '-i', input_path, '-c:v', 'libx264', '-preset', 'fast',
                    '-crf', '23', '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart',
                    '-y', output_path], check=True, capture_output=True)


def sample_frames_from_video(video_path, num_samples=10):
    if not video_path:
        raise ValueError("No input video provided")
    if not video_path.endswith('.mp4'):
        tmp_mp4 = tempfile.mktemp(suffix='.mp4')
        convert_video_to_mp4(video_path, tmp_mp4)
        video_path = tmp_mp4
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS)) or 5
    if total_frames == 0:
        raise ValueError("Unable to read video frames")
    if num_samples >= total_frames:
        frame_indices = list(range(total_frames))
    else:
        step = total_frames / num_samples
        frame_indices = [int(i * step) for i in range(num_samples)]

    def sharpness(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    os.makedirs(TMP_DIR, exist_ok=True)
    temp_dir = tempfile.mkdtemp(dir=TMP_DIR)
    frames, search_window = [], 3
    for i, target_idx in enumerate(frame_indices):
        best_frame, best_sharp = None, -1
        for f_idx in range(max(0, target_idx - search_window), min(total_frames - 1, target_idx + search_window) + 1):
            video.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ok, frame = video.read()
            if ok:
                s = sharpness(frame)
                if s > best_sharp:
                    best_sharp, best_frame = s, frame.copy()
        if best_frame is not None:
            frames.append(cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(temp_dir, f"{len(frames)-1:05d}.jpg"), best_frame)
    video.release()
    return frames, temp_dir, fps


# ── SAM2 dual-mask propagation + masked video ─────────────────────────────────
def propagate_masks_sam2(frames_dir, num_frames, object_points, hand_points):
    with torch.inference_mode():
        state = _SAM2.init_state(video_path=frames_dir)
        _SAM2.reset_state(state)
        H, W = state['video_height'], state['video_width']
        if object_points:
            _SAM2.add_new_points_or_box(
                inference_state=state, frame_idx=0, obj_id=OBJECT_OBJ_ID,
                points=np.array(object_points, dtype=np.float32),
                labels=np.ones(len(object_points), dtype=np.int32))
        if hand_points:
            _SAM2.add_new_points_or_box(
                inference_state=state, frame_idx=0, obj_id=HAND_OBJ_ID,
                points=np.array(hand_points, dtype=np.float32),
                labels=np.ones(len(hand_points), dtype=np.int32))
        obj_by_idx, hand_by_idx = {}, {}
        for f_idx, out_obj_ids, logits in _SAM2.propagate_in_video(state):
            obj_m = np.zeros((H, W), dtype=bool); hand_m = np.zeros((H, W), dtype=bool)
            for i, oid in enumerate(out_obj_ids):
                mm = (logits[i] > 0.0).cpu().numpy().squeeze()
                if oid == OBJECT_OBJ_ID:
                    obj_m = mm
                elif oid == HAND_OBJ_ID:
                    hand_m = mm
            obj_by_idx[f_idx] = obj_m; hand_by_idx[f_idx] = hand_m
    object_masks = [obj_by_idx.get(i, np.zeros((H, W), bool)) for i in range(num_frames)]
    hand_masks = [hand_by_idx.get(i, np.zeros((H, W), bool)) for i in range(num_frames)]
    return object_masks, hand_masks




# ── app state + point drawing ─────────────────────────────────────────────────
class AppState:
    def __init__(self):
        self.frames = []
        self.base_first_frame = None
        self.temp_dir = None
        self.fps = 5
        self.object_points = []
        self.hand_points = []
        self.object_masks = []
        self.hand_masks = []
        self.voxels = None

app_state = AppState()


def draw_points_on_frame(frame, object_points, hand_points):
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    for idx, p in enumerate(object_points):
        cv2.circle(bgr, p, 12, (255, 255, 255), -1); cv2.circle(bgr, p, 8, (0, 0, 255), -1)
        cv2.putText(bgr, "obj" + str(idx + 1), (p[0] + 15, p[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for idx, p in enumerate(hand_points):
        cv2.circle(bgr, p, 12, (255, 255, 255), -1); cv2.circle(bgr, p, 8, (0, 255, 0), -1)
        cv2.putText(bgr, "hand" + str(idx + 1), (p[0] + 15, p[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def get_marked_frame():
    if app_state.base_first_frame is None:
        return None
    return draw_points_on_frame(app_state.base_first_frame, app_state.object_points, app_state.hand_points)


def process_video_upload(video_path, num_samples=10):
    global app_state
    try:
        app_state = AppState()
        frames, temp_dir, fps = sample_frames_from_video(video_path, int(num_samples))
        app_state.frames = frames
        app_state.base_first_frame = frames[0].copy() if frames else None
        app_state.temp_dir = temp_dir
        app_state.fps = fps
        if not frames:
            return gr.update(visible=False), "No frames extracted from video", gr.update(), gr.update()
        return (gr.update(value=get_marked_frame(), visible=True),
                f"Extracted {len(frames)} frames. Click points on the first frame for object (red) and hand (green).",
                gr.update(visible=False), gr.update(visible=False))
    except Exception as e:
        return gr.update(visible=False), f"Error processing video: {e}", gr.update(), gr.update()


def handle_point_selection(mask_type, evt: gr.SelectData):
    x, y = evt.index[0], evt.index[1]
    if mask_type == "object":
        app_state.object_points.append((x, y))
        msg = f"Object point added at ({x}, {y}). Total object points: {len(app_state.object_points)}"
    elif mask_type == "hand":
        app_state.hand_points.append((x, y))
        msg = f"Hand point added at ({x}, {y}). Total hand points: {len(app_state.hand_points)}"
    else:
        return "Invalid mask type", get_marked_frame()
    return msg, get_marked_frame()


def clear_object_points():
    app_state.object_points = []
    return "Object points cleared", get_marked_frame()


def clear_hand_points():
    app_state.hand_points = []
    return "Hand points cleared", get_marked_frame()


def clear_all_points():
    app_state.object_points = []; app_state.hand_points = []
    return "All points cleared", get_marked_frame()


def generate_masks():
    """Run SAM2 propagation. Returns (ok, message)."""
    try:
        if not app_state.frames or not app_state.temp_dir:
            return False, "No frames available. Please upload a video first."
        if not app_state.object_points:
            return False, "Please mark at least one Object point on the first frame."
        object_masks, hand_masks = propagate_masks_sam2(
            app_state.temp_dir, len(app_state.frames),
            app_state.object_points, app_state.hand_points,
        )
        app_state.object_masks = object_masks
        app_state.hand_masks = hand_masks
        return True, "Masks generated."
    except Exception as e:
        return False, f"Error generating masks: {e}"


