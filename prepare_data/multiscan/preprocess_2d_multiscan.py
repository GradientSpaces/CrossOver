import os
import cv2
import json
import jsonlines
import argparse
import os.path as osp
import shutil

def process_scene_folders(base_dir, frame_interval=10):
    base_dir=osp.join(base_dir, 'scenes')
    scene_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for scene_folder in scene_folders:
        scene_path = os.path.join(base_dir, scene_folder)
        video_path = os.path.join(scene_path, f"{scene_folder}.mp4")
        jsonl_path = os.path.join(scene_path, f"{scene_folder}.jsonl")
        frame_output_dir = os.path.join(scene_path, "sequence")
        frame_ids_txt_path = os.path.join(scene_path, "frame_ids.txt")
        metadata_output_path = os.path.join(scene_path, "poses.jsonl")

        if os.path.exists(frame_output_dir):
            shutil.rmtree(frame_output_dir)
        os.makedirs(frame_output_dir)

        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            continue
        if not os.path.exists(jsonl_path):
            print(f"Metadata file not found: {jsonl_path}")
            continue

        print(f"Processing scene: {scene_folder}")

        frame_ids = extract_frames_from_video(video_path, frame_output_dir, frame_interval)

        with open(frame_ids_txt_path, "w") as f:
            for frame_id in frame_ids:
                f.write(f"{frame_id}\n")

        selected_metadata = extract_metadata_by_line_number(jsonl_path, frame_ids)

        with jsonlines.open(metadata_output_path, mode="w") as writer:
            for entry in selected_metadata:
                writer.write(entry)

        print(f"Finished processing scene: {scene_folder}")


def extract_frames_from_video(video_path, output_dir, frame_interval):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_ids = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        if frame_count % frame_interval == 0:
            frame_id = frame_count
            frame_ids.append(frame_id)
            output_path = os.path.join(output_dir, f"frame-{frame_id}.color.jpg")
            cv2.imwrite(output_path, frame)  # Save frame as an image

        frame_count += 1

    cap.release()
    return frame_ids


def extract_metadata_by_line_number(jsonl_path, line_numbers):

    selected_metadata = []

    with jsonlines.open(jsonl_path) as reader:
        for line_idx, entry in enumerate(reader):
            if line_idx in line_numbers:
                entry["frame_id"] = line_idx
                selected_metadata.append(entry)

    return selected_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process scene folders.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base dataset directory.")
    parser.add_argument("--frame_interval", type=int, default=10, help="Interval for saving frames.")
    args = parser.parse_args()

    process_scene_folders(args.base_dir, args.frame_interval)