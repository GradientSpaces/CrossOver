import os
import shutil

def move_annotation_files(source_base_folder, target_base_folder):
    # Iterate through scenes in the source folder
    for scene in os.listdir(source_base_folder):
        source_scene_path = os.path.join(source_base_folder, scene)
        target_scene_path = os.path.join(target_base_folder, scene)

        # Ensure the scene exists in both source and target
        if not os.path.isdir(source_scene_path) or not os.path.isdir(target_scene_path):
            continue

        # Check if annotation_3d.json exists in the source scene folder
        annotation_file = os.path.join(source_scene_path, "annotation_3d.json")
        if os.path.exists(annotation_file):
            target_annotation_file = os.path.join(target_scene_path, "annotation_3d.json")

            print(f"Moving {annotation_file} to {target_annotation_file}")
            shutil.move(annotation_file, target_annotation_file)
            os.remove(annotation_file)
        else:
            print(f"No annotation_3d.json found in {source_scene_path}. Skipping.")

if __name__ == "__main__":
    DIR_PREFIX='/Users/gauravpradeep/CrossOver_ScaleUp/'
    # Define the source and target base folder paths
    # source_base_folder = "Structured3D-1"  # Folder where annotation_3d.json currently exists
    target_base_folder = DIR_PREFIX+"Structured3D"  # Folder where it should be moved

    for folder in ["Structured3D-1", "Structured3D-2", "Structured3D-3", "Structured3D-4", "Structured3D-5", "Structured3D-6", "Structured3D-7", "Structured3D-8", "Structured3D-9", "Structured3D-11", "Structured3D-12", "Structured3D-13", "Structured3D-14", "Structured3D-15", "Structured3D-16", "Structured3D-17", "Structured3D-18"]:
        source_base_folder = DIR_PREFIX+folder
        move_annotation_files(source_base_folder, target_base_folder)

