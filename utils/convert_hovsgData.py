import os
import numpy as np
from PIL import Image

def convert_images_to_npy(dirr):
    depth_dir = os.path.join(dirr, 'depth')
    rgb_dir = os.path.join(dirr, 'rgb')

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
    scenes = ["office1","office2","office3","office4","room0","room1","room2"]
    for scene in scenes:
        dirr = f"/nvme0n1/hong/VLMAPS/InstanceSeemMap/Data/habitat_sim/Replica/{scene}/results"
        convert_images_to_npy(dirr)
        print(f"Conversion completed.: {scene} | {scenes.index(scene)+1}/{len(scenes)}")
