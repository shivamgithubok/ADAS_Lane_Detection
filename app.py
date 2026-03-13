import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from lane import process_video_frames, resolve_default_video_path

st.set_page_config(page_title="Lane Detection", layout="wide")


def build_roi_points(top_left_x, top_left_y, bottom_left_x, bottom_y, bottom_right_x, top_right_x, top_right_y):
    return np.float32(
        [
            (top_left_x, top_left_y),
            (bottom_left_x, bottom_y),
            (bottom_right_x, bottom_y),
            (top_right_x, top_right_y),
        ]
    )


def draw_roi_preview(frame, roi_points):
    preview = frame.copy()
    cv2.polylines(preview, [np.int32(roi_points)], True, (255, 0, 255), 3)
    return preview


def main():
    st.title("Lane Detection Dashboard")
    st.caption("Frontend preview using a video file instead of live streaming.")

    default_video = resolve_default_video_path()
    default_label = default_video.name if default_video else "None"

    st.sidebar.header("Playback")
    frame_delay = st.sidebar.slider("Frame delay (ms)", min_value=1, max_value=200, value=30, step=1)
    max_frames = st.sidebar.slider("Preview frames", min_value=30, max_value=1500, value=300, step=30)

    st.sidebar.write(f"Detected source: `{default_label}`")
    uploaded_video = st.sidebar.file_uploader("Optional video override", type=["mp4", "mov", "avi"])

    active_video_path = None
    temp_video_path = Path("video.mp4")

    if uploaded_video is not None:
        temp_video_path.write_bytes(uploaded_video.getbuffer())
        active_video_path = temp_video_path
        st.sidebar.success("Using uploaded file as video.mp4")
    else:
        active_video_path = default_video

    if active_video_path is None:
        st.error("No input video found. Add `video.mp4` or keep `project_video.mp4` in the project root.")
        return

    st.write(f"Running lane detection on `{Path(active_video_path).name}`")
    preview = cv2.VideoCapture(str(active_video_path))
    success, frame = preview.read()
    preview.release()
    if not success:
        st.error("Unable to read the selected video.")
        return

    frame_height, frame_width = frame.shape[:2]
    st.sidebar.header("Lane ROI")
    st.sidebar.caption("Move these points until the trapezoid sits on the road lanes.")
    top_left_x = st.sidebar.slider("Top-left X", min_value=0, max_value=frame_width - 1, value=min(274, frame_width - 1))
    top_left_y = st.sidebar.slider("Top-left Y", min_value=0, max_value=frame_height - 1, value=min(184, frame_height - 1))
    top_right_x = st.sidebar.slider("Top-right X", min_value=0, max_value=frame_width - 1, value=min(371, frame_width - 1))
    top_right_y = st.sidebar.slider("Top-right Y", min_value=0, max_value=frame_height - 1, value=min(184, frame_height - 1))
    bottom_left_x = st.sidebar.slider("Bottom-left X", min_value=0, max_value=frame_width - 1, value=0)
    bottom_right_x = st.sidebar.slider("Bottom-right X", min_value=0, max_value=frame_width - 1, value=min(575, frame_width - 1))
    bottom_y = st.sidebar.slider("Bottom Y", min_value=0, max_value=frame_height - 1, value=min(337, frame_height - 1))

    roi_points = build_roi_points(
        top_left_x,
        top_left_y,
        bottom_left_x,
        bottom_y,
        bottom_right_x,
        top_right_x,
        top_right_y,
    )

    start = st.button("Start Lane Detection", type="primary")

    original_col, processed_col = st.columns(2)
    original_slot = original_col.empty()
    processed_slot = processed_col.empty()
    status = st.empty()

    if not start:
        original_slot.image(
            cv2.cvtColor(draw_roi_preview(frame, roi_points), cv2.COLOR_BGR2RGB),
            caption="ROI Preview",
            use_container_width=True,
        )
        processed_slot.info("Press 'Start Lane Detection' to process the video.")
        return

    processed_count = 0
    started_at = time.time()
    for original_frame, processed_frame in process_video_frames(
        video_path=active_video_path,
        max_frames=max_frames,
        roi_points=roi_points,
    ):
        original_slot.image(
            cv2.cvtColor(draw_roi_preview(original_frame, roi_points), cv2.COLOR_BGR2RGB),
            caption="Original Video With ROI",
            use_container_width=True,
        )
        processed_slot.image(
            cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
            caption="Lane Detection Output",
            use_container_width=True,
        )
        processed_count += 1
        elapsed = max(time.time() - started_at, 0.001)
        status.write(f"Processed frames: {processed_count} | Approx. FPS: {processed_count / elapsed:.2f}")
        time.sleep(frame_delay / 1000)

    st.success("Video preview completed.")


if __name__ == "__main__":
    main()
