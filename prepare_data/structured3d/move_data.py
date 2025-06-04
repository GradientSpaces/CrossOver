import os
import shutil

def combine_files(base_folder_1, base_folder_2):
    count = 0
    # Iterate through scenes in the first base folder
    for scene in os.listdir(base_folder_1):
        scene_path_1 = os.path.join(base_folder_1, scene)
        scene_path_2 = os.path.join(base_folder_2, scene)  # Corresponding scene in the second base folder

        if not os.path.isdir(scene_path_1) or not os.path.isdir(scene_path_2):
            continue

        rendering_path_1 = os.path.join(scene_path_1, "2D_rendering")
        rendering_path_2 = os.path.join(scene_path_2, "2D_rendering")

        if not os.path.exists(rendering_path_1) or not os.path.exists(rendering_path_2):
            continue

        # Iterate through render IDs
        for render_id in os.listdir(rendering_path_1):
            render_path_1 = os.path.join(rendering_path_1, render_id)
            render_path_2 = os.path.join(rendering_path_2, render_id)

            perspective_path_1 = os.path.join(render_path_1, "perspective", "full")
            perspective_path_2 = os.path.join(render_path_2, "perspective", "full")
            
            if not os.path.exists(perspective_path_1) or not os.path.exists(perspective_path_2):
                continue

            # Iterate through views in the perspective folder
            for view in os.listdir(perspective_path_1):
                view_path_1 = os.path.join(perspective_path_1, view)
                view_path_2 = os.path.join(perspective_path_2, view)

                if not os.path.isdir(view_path_1) or not os.path.isdir(view_path_2):
                    continue

                # Check if instance.png exists in the target perspective folder
                instance_file = os.path.join(view_path_2, "instance.png")
                if not os.path.exists(instance_file):
                    print(f"No instance.png found in {view_path_2}. Skipping.")
                    continue

                # Move all files from the source folder to the target folder
                for file_name in os.listdir(view_path_1):
                    source_file = os.path.join(view_path_1, file_name)
                    target_file = os.path.join(view_path_2, file_name)

                    if os.path.isfile(source_file):
                        print(f"Moving {source_file} to {target_file}")
                        shutil.move(source_file, target_file)

                # Optionally: Remove the now-empty view folder
                if not os.listdir(view_path_1):
                    print(f"Removing empty folder: {view_path_1}")
                    os.rmdir(view_path_1)
        count += 1
    print(f"Processed {count} scenes.")

if __name__ == "__main__":
    DIR_PREFIX='/Users/gauravpradeep/CrossOver_ScaleUp/'
    base_folder_2 = DIR_PREFIX+"Structured3D"  # Directory with perspective instance.png
    # base_folder_1 = "Structured3D-1"  # Directory with albedo.png, etc.
    
    for folder in ["Structured3D-1", "Structured3D-2", "Structured3D-3", "Structured3D-4", "Structured3D-5", "Structured3D-6", "Structured3D-7", "Structured3D-8", "Structured3D-9", "Structured3D-11", "Structured3D-12", "Structured3D-13", "Structured3D-14", "Structured3D-15", "Structured3D-16", "Structured3D-17", "Structured3D-18"]:
        base_folder_1 = DIR_PREFIX+folder
        combine_files(base_folder_1, base_folder_2)
    
