import os
import numpy as np
from PIL import Image

def convert_images_to_npy(scene_id, dirr):
    depth_dir = os.path.join(dirr, 'depth')
    depth_dir2 = os.path.join(dirr, 'depth2')
    rgb_dir = os.path.join(dirr, 'rgb')
    rgb_dir2 = os.path.join(dirr, 'rgb2')
    pose_dir = os.path.join(dirr, 'pose')
    pose_dir2 = os.path.join(dirr, 'pose2')
    semantic_dir = os.path.join(dirr, 'semantic')
    semantic_dir2 = os.path.join(dirr, 'semantic2')

    # Create directories if they don't exist
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(depth_dir2, exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(rgb_dir2, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)
    os.makedirs(pose_dir2, exist_ok=True)
    os.makedirs(semantic_dir, exist_ok=True)
    os.makedirs(semantic_dir2, exist_ok=True)

    scene = scene_id.split("-")[-1]
    depth_files = [f for f in os.listdir(depth_dir) if f.endswith(".png") and f.startswith(scene)]
    num_files = len(depth_files)
    for i in range(num_files):
        file_index = f"{i:06d}"

        # Process depth files
        depth_filename = os.path.join(depth_dir, f"{scene}_{file_index}.png")
        if os.path.exists(depth_filename):
            depth_image = Image.open(depth_filename)
            depth_array = np.array(depth_image)
            np.save(os.path.join(depth_dir2, f"{file_index}.npy"), depth_array)
        else:
            print(f"Depth file not found: {depth_filename}")

        # Process frame (RGB) files
        frame_filename = os.path.join(rgb_dir, f"{scene}_{file_index}.png")
        if os.path.exists(frame_filename):
            frame_image = Image.open(frame_filename)
            frame_image.save(os.path.join(rgb_dir2, f"{file_index}.png"))
        else:
            print(f"Frame file not found: {frame_filename}")

        # Process pose files
        frame_filename = os.path.join(pose_dir, f"{scene}_{file_index}.txt")
        if os.path.exists(frame_filename):
            with open(frame_filename, "r") as f_in:
                content = f_in.read()
            with open(os.path.join(pose_dir2, f"{file_index}.txt"), "w") as f_out:
                f_out.write(content)
        else:
            print(f"Frame file not found: {frame_filename}")
        
        # Process semantic files
        semantic_filename = os.path.join(semantic_dir, f"{scene}_{file_index}.npy")
        if os.path.exists(semantic_filename):
            semantic_array = np.load(semantic_filename)
            np.save(os.path.join(semantic_dir2, f"{file_index}.npy"), semantic_array)
        else:
            print(f"Semantic file not found: {semantic_filename}")

if __name__ == "__main__":
    # scenes = ["00824-Dd4bFSTQ8gi"]#["office1","office2","office3","office4","room0","room1","room2"]
    scenes = ["00862-LT9Jq6dN3Ea","00873-bxsVRursffK","00843-DYehNKdT76V","00861-GLAQ4DNUx5U","00877-4ok3usBNeis","00890-6s7QHgap2fW"]#,"00862-LT9Jq6dN3Ea"]
    for scene in scenes:
        dirr = f"/home/vlmap_RCI/Data/habitat_sim/hm3dsem/{scene}/"
        convert_images_to_npy(scene, dirr)
        print(f"Conversion completed.: {scene} | {scenes.index(scene)+1}/{len(scenes)}")
