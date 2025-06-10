import os
import shutil
import os.path as osp
def combine_files(in_folder, out_folder):
    count = 0
    # Iterate through scenes in the first base folder
    for scene in os.listdir(in_folder):
        scene_path_in = os.path.join(in_folder, scene)
        scene_path_out = os.path.join(out_folder, scene)  # Corresponding scene in the second base folder
        print(f"Processing scene: {scene_path_in} -> {scene_path_out}")
        if not os.path.isdir(scene_path_in) or not os.path.isdir(scene_path_out):
            continue
        annotation_file_in = os.path.join(scene_path_in, "annotation_3d.json")
        annotation_file_out = os.path.join(scene_path_out, "annotation_3d.json")
        # print(f"Annotation file in: {annotation_file_in}, Annotation file out: {annotation_file_out}")
        shutil.move(annotation_file_in, annotation_file_out)
        print(f"Moved {annotation_file_in} to {annotation_file_out}")

        rendering_path_in = os.path.join(scene_path_in, "2D_rendering")
        rendering_path_out = os.path.join(scene_path_out, "2D_rendering")
        print(f"Rendering path in: {rendering_path_in}, Rendering path out: {rendering_path_out}")
        if not os.path.exists(rendering_path_in) or not os.path.exists(rendering_path_out):
            continue

        # Iterate through render IDs
        for render_id in os.listdir(rendering_path_in):
            print("hello")
            render_path_in = os.path.join(rendering_path_in, render_id)
            render_path_out = os.path.join(rendering_path_out, render_id)

            perspective_path_in = os.path.join(render_path_in, "perspective", "full")
            perspective_path_out = os.path.join(render_path_out, "perspective", "full")
            print(f"Perspective path in: {perspective_path_in}, Perspective path out: {perspective_path_out}")
            
            if not os.path.exists(perspective_path_in) or not os.path.exists(perspective_path_out):
                continue
            print(f"Processing render ID: {render_id} -> {render_path_in} -> {render_path_out}")
            # Iterate through views in the perspective folder
            for view in os.listdir(perspective_path_in):
                view_path_in = os.path.join(perspective_path_in, view)
                view_path_out = os.path.join(perspective_path_out, view)
                print(f"Processing view: {view_path_in} -> {view_path_out}")

                if not os.path.isdir(view_path_in) or not os.path.isdir(view_path_out):
                    continue

                # Check if instance.png exists in the target perspective folder
                instance_file = os.path.join(view_path_out, "instance.png")
                if not os.path.exists(instance_file):
                    print(f"No instance.png found in {view_path_out}. Skipping.")
                    continue

                # Move all files from the source folder to the target folder
                for file_name in os.listdir(view_path_in):
                    source_file = os.path.join(view_path_in, file_name)
                    target_file = os.path.join(view_path_out, file_name)

                    if os.path.isfile(source_file):
                        print(f"Moving {source_file} to {target_file}")
                        shutil.move(source_file, target_file)

                # Optionally: Remove the now-empty view folder
                if not os.listdir(view_path_in):
                    print(f"Removing empty folder: {view_path_in}")
                    os.rmdir(view_path_in)
        count += 1
    print(f"Processed {count} scenes.")

if __name__ == "__main__":
    EXTRACTED_DIR='/Users/gauravpradeep/CrossOver_ScaleUp/extracted'
    out_dir = osp.join(EXTRACTED_DIR,"Structured3D_bbox/Structured3D")  # this dir is the one that has perspective instance.png and bbox3d.json
    print(out_dir)
    for folder in os.listdir(EXTRACTED_DIR):
        if folder == 'Structured3D_bbox':
            continue
        in_folder = osp.join(EXTRACTED_DIR, folder, "Structured3D")
        print(f"Processing folder: {in_folder}")
        combine_files(in_folder, out_dir)
    
