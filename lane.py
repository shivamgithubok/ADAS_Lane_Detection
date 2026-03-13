from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

import edge_detection as edge


def resolve_default_video_path() -> Path | None:
    for candidate in ("video.mp4","project_video.mp4", "video.avi", "video.mov"):
        path = Path(candidate)
        if path.exists():
            return path
    return None


def default_roi_points(frame_width: int, frame_height: int) -> np.ndarray:
    return np.float32([
        (int(frame_width * 0.46), int(frame_height * 0.63)),  # top-left  (narrower apex)
        (int(frame_width * 0.14), int(frame_height * 0.95)),  # bottom-left
        (int(frame_width * 0.88), int(frame_height * 0.95)),  # bottom-right
        (int(frame_width * 0.56), int(frame_height * 0.63)),  # top-right
    ])


class Lane:
    def __init__(self, orig_frame: np.ndarray, roi_points: np.ndarray | None = None):
        if orig_frame is None or orig_frame.size == 0:
            raise ValueError("orig_frame must be a valid image frame")

        self.orig_frame = orig_frame
        self.height, self.width = self.orig_frame.shape[:2]
        self.orig_image_size = (self.width, self.height)

        self.roi_points = (
            np.float32(roi_points)
            if roi_points is not None
            else default_roi_points(self.width, self.height)
        )
        self.padding = int(0.25 * self.width)
        self.desired_roi_points = np.float32(
            [
                [self.padding, 0],
                [self.padding, self.height],
                [self.width - self.padding, self.height],
                [self.width - self.padding, 0],
            ]
        )

        self.lane_line_markings: np.ndarray | None = None
        self.warped_frame: np.ndarray | None = None
        self.transformation_matrix: np.ndarray | None = None
        self.inv_transformation_matrix: np.ndarray | None = None
        self.histogram: np.ndarray | None = None

        self.no_of_windows = 10
        self.margin = max(40, int(self.width / 12))
        self.minpix = max(20, int(self.width / 24))

        self.left_fit: np.ndarray | None = None
        self.right_fit: np.ndarray | None = None
        self.left_lane_inds = None
        self.right_lane_inds = None
        self.ploty: np.ndarray | None = None
        self.left_fitx: np.ndarray | None = None
        self.right_fitx: np.ndarray | None = None
        self.leftx: np.ndarray | None = None
        self.rightx: np.ndarray | None = None
        self.lefty: np.ndarray | None = None
        self.righty: np.ndarray | None = None

        self.YM_PER_PIX = 10.0 / 1000
        self.XM_PER_PIX = 3.7 / 781

        self.left_curvem: float | None = None
        self.right_curvem: float | None = None
        self.center_offset: float | None = None

    def get_line_markings(self, frame: np.ndarray | None = None) -> np.ndarray:
        frame = self.orig_frame if frame is None else frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lightness = hls[:, :, 1]
        _, bright_binary = edge.threshold(lightness, thresh=(150, 255))
        _, white_binary = edge.threshold(gray, thresh=(170, 255))
        white_mask = cv2.bitwise_and(bright_binary, white_binary)

        yellow_mask = cv2.inRange(
            hsv,
            np.array([12, 70, 70], dtype=np.uint8),
            np.array([40, 255, 255], dtype=np.uint8),
        )

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * sobelx / max(np.max(sobelx), 1))
        _, gradient_binary = edge.threshold(scaled_sobel, thresh=(25, 255))
        gradient_mask = cv2.bitwise_and(gradient_binary, cv2.bitwise_or(bright_binary, yellow_mask))

        combined_mask = cv2.bitwise_or(cv2.bitwise_or(white_mask, yellow_mask), gradient_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.lane_line_markings = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        return self.lane_line_markings

    def perspective_transform(self, frame: np.ndarray | None = None) -> np.ndarray:
        frame = self.lane_line_markings if frame is None else frame
        if frame is None:
            raise ValueError("lane line markings must be created before perspective transform")

        self.transformation_matrix = cv2.getPerspectiveTransform(
            self.roi_points, self.desired_roi_points
        )
        self.inv_transformation_matrix = cv2.getPerspectiveTransform(
            self.desired_roi_points, self.roi_points
        )
        self.warped_frame = cv2.warpPerspective(
            frame,
            self.transformation_matrix,
            self.orig_image_size,
            flags=cv2.INTER_LINEAR,
        )
        _, binary_warped = cv2.threshold(self.warped_frame, 127, 255, cv2.THRESH_BINARY)
        self.warped_frame = binary_warped
        return self.warped_frame

    def calculate_histogram(self, frame: np.ndarray | None = None) -> np.ndarray:
        frame = self.warped_frame if frame is None else frame
        if frame is None:
            raise ValueError("warped frame must exist before histogram calculation")

        self.histogram = np.sum(frame[frame.shape[0] // 2 :, :], axis=0)
        return self.histogram

    def histogram_peak(self) -> tuple[int, int]:
        if self.histogram is None:
            raise ValueError("histogram must be calculated first")

        midpoint = self.histogram.shape[0] // 2
        leftx_base = int(np.argmax(self.histogram[:midpoint]))
        rightx_base = int(np.argmax(self.histogram[midpoint:]) + midpoint)
        return leftx_base, rightx_base

    def _fit_from_lane_indices(self, left_lane_inds, right_lane_inds) -> tuple[np.ndarray, np.ndarray]:
        nonzero = self.warped_frame.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        if len(leftx) < 3 or len(rightx) < 3:
            raise ValueError("insufficient lane pixels detected")

        self.leftx = leftx
        self.lefty = lefty
        self.rightx = rightx
        self.righty = righty

        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
        self.left_lane_inds = left_lane_inds
        self.right_lane_inds = right_lane_inds
        self._update_plot_points()
        return self.left_fit, self.right_fit

    def _update_plot_points(self) -> None:
        self.ploty = np.linspace(0, self.height - 1, self.height)
        self.left_fitx = self.left_fit[0] * self.ploty**2 + self.left_fit[1] * self.ploty + self.left_fit[2]
        self.right_fitx = self.right_fit[0] * self.ploty**2 + self.right_fit[1] * self.ploty + self.right_fit[2]

    def get_lane_line_indices_sliding_windows(self) -> tuple[np.ndarray, np.ndarray]:
        if self.warped_frame is None:
            raise ValueError("warped frame must exist before sliding window search")

        window_height = self.warped_frame.shape[0] // self.no_of_windows
        nonzero = self.warped_frame.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = []
        right_lane_inds = []
        leftx_current, rightx_current = self.histogram_peak()

        for window in range(self.no_of_windows):
            win_y_low = self.warped_frame.shape[0] - (window + 1) * window_height
            win_y_high = self.warped_frame.shape[0] - window * window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            good_left_inds = (
                (
                    (nonzeroy >= win_y_low)
                    & (nonzeroy < win_y_high)
                    & (nonzerox >= win_xleft_low)
                    & (nonzerox < win_xleft_high)
                )
                .nonzero()[0]
            )
            good_right_inds = (
                (
                    (nonzeroy >= win_y_low)
                    & (nonzeroy < win_y_high)
                    & (nonzerox >= win_xright_low)
                    & (nonzerox < win_xright_high)
                )
                .nonzero()[0]
            )

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > self.minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        return self._fit_from_lane_indices(np.concatenate(left_lane_inds), np.concatenate(right_lane_inds))

    def get_lane_line_previous_window(
        self, left_fit: np.ndarray, right_fit: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.warped_frame is None:
            raise ValueError("warped frame must exist before previous-window search")

        nonzero = self.warped_frame.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = (
            (
                nonzerox
                > (left_fit[0] * nonzeroy**2 + left_fit[1] * nonzeroy + left_fit[2] - self.margin)
            )
            & (
                nonzerox
                < (left_fit[0] * nonzeroy**2 + left_fit[1] * nonzeroy + left_fit[2] + self.margin)
            )
        )
        right_lane_inds = (
            (
                nonzerox
                > (right_fit[0] * nonzeroy**2 + right_fit[1] * nonzeroy + right_fit[2] - self.margin)
            )
            & (
                nonzerox
                < (right_fit[0] * nonzeroy**2 + right_fit[1] * nonzeroy + right_fit[2] + self.margin)
            )
        )

        return self._fit_from_lane_indices(left_lane_inds, right_lane_inds)

    def seed_from_previous_fit(self, left_fit: np.ndarray, right_fit: np.ndarray) -> None:
        self.left_fit = left_fit
        self.right_fit = right_fit
        self._update_plot_points()

    def overlay_lane_lines(self) -> np.ndarray:
        if self.warped_frame is None or self.inv_transformation_matrix is None:
            raise ValueError("perspective transform must run before overlay")
        if self.left_fitx is None or self.right_fitx is None or self.ploty is None:
            raise ValueError("lane fit must exist before overlay")

        warp_zero = np.zeros_like(self.warped_frame).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))
        newwarp = cv2.warpPerspective(
            color_warp,
            self.inv_transformation_matrix,
            (self.orig_frame.shape[1], self.orig_frame.shape[0]),
        )
        return cv2.addWeighted(self.orig_frame, 1, newwarp, 0.3, 0)

    def calculate_curvature(self) -> tuple[float, float]:
        if self.ploty is None or self.leftx is None or self.rightx is None:
            raise ValueError("lane pixels must exist before curvature calculation")

        y_eval = np.max(self.ploty)
        left_fit_cr = np.polyfit(self.lefty * self.YM_PER_PIX, self.leftx * self.XM_PER_PIX, 2)
        right_fit_cr = np.polyfit(self.righty * self.YM_PER_PIX, self.rightx * self.XM_PER_PIX, 2)

        self.left_curvem = ((1 + (2 * left_fit_cr[0] * y_eval * self.YM_PER_PIX + left_fit_cr[1]) ** 2) ** 1.5) / max(
            np.absolute(2 * left_fit_cr[0]),
            1e-6,
        )
        self.right_curvem = (
            (1 + (2 * right_fit_cr[0] * y_eval * self.YM_PER_PIX + right_fit_cr[1]) ** 2) ** 1.5
        ) / max(np.absolute(2 * right_fit_cr[0]), 1e-6)

        return self.left_curvem, self.right_curvem

    def calculate_car_position(self) -> float:
        if self.left_fit is None or self.right_fit is None:
            raise ValueError("lane fit must exist before offset calculation")

        car_location = self.orig_frame.shape[1] / 2
        height = self.orig_frame.shape[0]
        bottom_left = self.left_fit[0] * height**2 + self.left_fit[1] * height + self.left_fit[2]
        bottom_right = self.right_fit[0] * height**2 + self.right_fit[1] * height + self.right_fit[2]

        center_lane = (bottom_right - bottom_left) / 2 + bottom_left
        self.center_offset = (np.abs(car_location) - np.abs(center_lane)) * self.XM_PER_PIX * 100
        return self.center_offset

    def display_curvature_offset(self, frame: np.ndarray | None = None) -> np.ndarray:
        image_copy = self.orig_frame.copy() if frame is None else frame.copy()

        curve_text = "Curve Radius: N/A"
        if self.left_curvem is not None and self.right_curvem is not None:
            curve_text = f"Curve Radius: {((self.left_curvem + self.right_curvem) / 2):.2f} m"

        offset_text = "Center Offset: N/A"
        if self.center_offset is not None:
            offset_text = f"Center Offset: {self.center_offset:.2f} cm"

        cv2.putText(image_copy, curve_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image_copy, offset_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return image_copy


class LaneVideoProcessor:
    def __init__(self, roi_points: np.ndarray | None = None):
        self.roi_points = np.float32(roi_points) if roi_points is not None else None
        self.previous_left_fit: np.ndarray | None = None
        self.previous_right_fit: np.ndarray | None = None
        self.previous_curvature: tuple[float, float] | None = None
        self.previous_center_offset: float | None = None

    def process(self, frame: np.ndarray) -> np.ndarray:
        lane = Lane(frame, roi_points=self.roi_points)
        edge_mask = lane.get_line_markings()
        lane.perspective_transform(edge_mask)
        lane.calculate_histogram()

        detected = False
        if self.previous_left_fit is not None and self.previous_right_fit is not None:
            try:
                lane.get_lane_line_previous_window(self.previous_left_fit, self.previous_right_fit)
                detected = True
            except (TypeError, ValueError, np.linalg.LinAlgError):
                detected = False

        if not detected:
            try:
                lane.get_lane_line_indices_sliding_windows()
                detected = True
            except (ValueError, np.linalg.LinAlgError):
                detected = False

        output_frame = frame.copy()
        if detected:
            self.previous_left_fit = lane.left_fit.copy()
            self.previous_right_fit = lane.right_fit.copy()
            output_frame = lane.overlay_lane_lines()

            try:
                self.previous_curvature = lane.calculate_curvature()
            except (ValueError, np.linalg.LinAlgError):
                self.previous_curvature = None

            try:
                self.previous_center_offset = lane.calculate_car_position()
            except ValueError:
                self.previous_center_offset = None

            output_frame = lane.display_curvature_offset(output_frame)
        elif self.previous_left_fit is not None and self.previous_right_fit is not None:
            lane.seed_from_previous_fit(self.previous_left_fit, self.previous_right_fit)
            output_frame = lane.overlay_lane_lines()
            lane.left_curvem = self.previous_curvature[0] if self.previous_curvature else None
            lane.right_curvem = self.previous_curvature[1] if self.previous_curvature else None
            lane.center_offset = self.previous_center_offset
            output_frame = lane.display_curvature_offset(output_frame)

        return edge.overlay_edge_preview(output_frame, edge_mask)


def process_video_frames(
    video_path: str | Path,
    max_frames: int | None = None,
    roi_points: np.ndarray | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    processor = LaneVideoProcessor(roi_points=roi_points)
    processed_frames = 0

    try:
        while True:
            if max_frames is not None and processed_frames >= max_frames:
                break

            success, frame = capture.read()
            if not success:
                break

            yield frame, processor.process(frame)
            processed_frames += 1
    finally:
        capture.release()


def main() -> None:
    video_path = resolve_default_video_path()
    if video_path is None:
        print("No video found. Add video.mp4 or project_video.mp4 to the project root.")
        return

    for index, _ in enumerate(process_video_frames(video_path=video_path, max_frames=5), start=1):
        print(f"Processed frame {index}")


if __name__ == "__main__":
    main()
