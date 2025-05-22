import os
import numpy as np
from PIL import Image
def poses2pose(poses_file, pose_dir):
    with open(poses_file, "r") as file:
        poses = file.readlines()
    os.makedirs(pose_dir, exist_ok=True)
    for idx, data in enumerate(poses):
        pose_filename = f"{idx:06}.txt"
        pose_filepath = os.path.join(pose_dir, pose_filename)
        with open(pose_filepath, "w") as pose_file:
            pose_file.write(data)

def convert_images_to_npy(dirr, path):
    depth_dir = os.path.join(path, 'depth')
    rgb_dir = os.path.join(path, 'rgb')

    # Create directories if they don't exist
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)

    for i in range(2000):
        file_index = f"{i:06d}"

        # Process depth files
        depth_filename = os.path.join(dirr, f"depth{file_index}.png")
        if os.path.exists(depth_filename):
            depth_image = Image.open(depth_filename)
            depth_array = np.array(depth_image)
            np.save(os.path.join(depth_dir, f"{file_index}.npy"), depth_array)
        else:
            print(f"Depth file not found: {depth_filename}")

        # Process frame (RGB) files
        frame_filename = os.path.join(dirr, f"frame{file_index}.jpg")
        if os.path.exists(frame_filename):
            frame_image = Image.open(frame_filename)
            frame_image.save(os.path.join(rgb_dir, f"{file_index}.png"))
        else:
            print(f"Frame file not found: {frame_filename}")

if __name__ == "__main__":
    path = "/nvme0n1/hong/VLMAPS/InstanceSeemMap/Data/habitat_sim/Replica/"
    scenes = ["office0", "office1","office2","office3","office4","room0","room1","room2"]
    for scene in scenes:
        img_dir = os.path.join(path, scene, "results")
        poses_dir = os.path.join(path, scene, "traj.txt")
        pose_dir = os.path.join(path, scene, "pose")
        # convert_images_to_npy(img_dir, path)
        poses2pose(poses_dir, pose_dir)
        print(f"Conversion completed.: {scene} | {scenes.index(scene)+1}/{len(scenes)}")
