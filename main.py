from pathlib import Path

from lane import process_video_frames, resolve_default_video_path


def main():
    video_path = resolve_default_video_path()
    if video_path is None:
        print("No video found. Add video.mp4 or project_video.mp4 to the project root.")
        return

    print(f"Processing preview from {Path(video_path).name}")
    for index, _ in enumerate(process_video_frames(video_path=video_path, max_frames=5), start=1):
        print(f"Processed frame {index}")


if __name__ == "__main__":
    main()
