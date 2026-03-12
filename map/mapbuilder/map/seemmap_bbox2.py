import numpy as np
from tqdm import tqdm
from collections import Counter

from map.mapbuilder.map.seemmap_bbox import SeemMap_bbox


class SeemMap_bbox2(SeemMap_bbox):
    """
    Experimental bbox path for performance comparison.
    Gate 4 optimization scope:
    - contact-cell based rgb mask fill
    - touched-cell based merge path in postprocessing2
    """

    def _build_instance_rgb_masks_cropped(self) -> None:
        if self.rot_map:
            grid_for_fill = np.rot90(self.grid, k=1) if self.hm3dsem_mat_mode else self.grid.T
        else:
            grid_for_fill = self.grid
        active_instance_pixels = {}

        for inst_id, inst in self.instance_dict.items():
            m = inst.get("mask")
            if m is None or np.sum(m) == 0:
                inst.pop("rgb_mask", None)
                inst.pop("rgb_bbox", None)
                continue

            ys, xs = np.where(m > 0)
            y0, x0, y1, x1 = ys.min(), xs.min(), ys.max() + 1, xs.max() + 1
            inst["rgb_bbox"] = (y0, x0, y1, x1)
            inst["rgb_mask"] = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
            active_instance_pixels[inst_id] = (ys, xs)

        for inst_id, (ys, xs) in active_instance_pixels.items():
            if inst_id in [1, 2]:
                continue
            inst = self.instance_dict.get(inst_id)
            if not inst or "rgb_mask" not in inst:
                continue

            y0, x0, _, _ = inst["rgb_bbox"]
            rgb_mask = inst["rgb_mask"]
            for y, x in zip(ys, xs):
                cell = grid_for_fill[y, x]
                if not cell:
                    continue
                val = cell.get(inst_id, None)
                if val is None or not (isinstance(val, (list, tuple)) and len(val) >= 4):
                    continue
                packed = val[3]
                if packed is None:
                    raise ValueError(f"packed RGB is None at ({y}, {x}), inst_id={inst_id}")
                if isinstance(packed, np.integer):
                    packed = int(packed)
                elif isinstance(packed, bool) or not isinstance(packed, int):
                    raise TypeError(f"invalid packed RGB type: {type(packed)} at ({y}, {x}), inst_id={inst_id}")
                if packed < 0 or packed > 0xFFFFFF:
                    raise ValueError(f"packed RGB out of range: {packed} at ({y}, {x}), inst_id={inst_id}")
                rgb_mask[y - y0, x - x0, :] = self._unpack_rgb(packed)

    def postprocessing2(self):
        init = True
        while True:
            if init:
                tsp = 0.9
            else:
                tsp = self.threshold_semSim_post

            grid_map = {}
            touched_cells = []
            for y in range(self.gs):
                for x in range(self.gs):
                    cell = self.grid[y, x]
                    if not cell:
                        continue
                    touched_cells.append((y, x, cell))
                    for key in cell.keys():
                        if key not in grid_map:
                            grid_map[key] = set()
                        grid_map[key].add((y, x))

            new_instance_dict = {}
            matching_dict = {}
            new_grid = np.empty((self.gs, self.gs), dtype=object)
            for i in range(self.gs):
                for j in range(self.gs):
                    new_grid[i, j] = None

            count0 = sum(1 for _, _, cell in touched_cells if 1 in cell)
            count1 = sum(1 for _, _, cell in touched_cells if 2 in cell)
            print(f"Size of wall and floor: {count0}, {count1}")

            pbar2 = tqdm(total=len(self.instance_dict.items()), leave=True)
            updated = False
            for instance_id, instance_val in self.instance_dict.items():
                tf = True
                coords = grid_map.get(instance_id, None)
                if coords is None:
                    pbar2.update(1)
                    continue
                instance_y, instance_x = zip(*coords)
                instance_y = np.array(instance_y)
                instance_x = np.array(instance_x)
                instance_mask = np.zeros((self.gs, self.gs), dtype=np.uint8)
                instance_mask[instance_y, instance_x] = 1
                bbox1 = self.calculate_bbox(instance_mask)
                instance_emb = instance_val["embedding"]
                instance_size = np.sum(instance_mask)

                for new_id, new_val in new_instance_dict.items():
                    new_mask = new_val["mask"]
                    new_emb = new_val["embedding"]
                    new_count = new_val["count"]
                    new_avg_height = new_val["avg_height"]

                    bbox2 = self.calculate_bbox(new_mask)
                    iou1 = self.calculate_geoSim(bbox1, bbox2)
                    iou2 = self.calculate_geoSim(bbox2, bbox1)
                    instance_emb_normalized = self._safe_normalize(instance_emb)
                    new_emb_normalized = self._safe_normalize(new_emb)
                    if instance_emb_normalized is None or new_emb_normalized is None:
                        continue
                    semSim = instance_emb_normalized @ new_emb_normalized.T
                    if max(iou1, iou2) > self.threshold_geoSim_post and semSim > tsp:
                        new_size = np.sum(new_mask)
                        i_ratio = instance_size / (instance_size + new_size)
                        n_ratio = new_size / (instance_size + new_size)
                        if not self.bool_size:
                            new_instance_dict[new_id]["embedding"] = (new_emb * new_count + instance_emb) / (new_count + 1)
                        else:
                            new_instance_dict[new_id]["embedding"] = instance_emb * i_ratio + new_emb * n_ratio
                        new_instance_dict[new_id]["avg_height"] = (new_avg_height * new_count + instance_val["avg_height"]) / (new_count + 1)
                        new_instance_dict[new_id]["count"] = new_count + 1
                        new_instance_dict[new_id]["size"] = new_size + instance_size
                        new_instance_dict[new_id]["mask"] = np.logical_or(new_mask, instance_mask).astype(np.uint8)
                        new_instance_dict[new_id]["frames"] = dict(Counter(new_instance_dict[new_id]["frames"]) + Counter(instance_val["frames"]))
                        tf = False
                        matching_dict[instance_id] = new_id
                        for frame_key in instance_val["frames"].keys():
                            frame_mask = self.frame_mask_dict[frame_key]
                            self.frame_mask_dict[frame_key][frame_mask == instance_id] = new_id
                        updated = True
                        break
                if tf:
                    if np.sum(instance_mask) < self.threshold_pixelSize_post:
                        pbar2.update(1)
                        continue
                    new_instance_dict[instance_id] = {
                        "mask": instance_mask,
                        "embedding": instance_emb,
                        "count": 1,
                        "size": instance_size,
                        "frames": instance_val["frames"],
                        "avg_height": instance_val["avg_height"],
                    }
                    matching_dict[instance_id] = instance_id
                pbar2.update(1)

            for instance_id in new_instance_dict.keys():
                frames = new_instance_dict[instance_id]["frames"]
                new_instance_dict[instance_id]["frames"] = dict(sorted(frames.items(), key=lambda x: x[1], reverse=True))
            print(new_instance_dict.keys())

            for y, x, cell in touched_cells:
                if new_grid[y, x] is None:
                    new_grid[y, x] = {}
                for key, val in cell.items():
                    if key in [1, 2]:
                        new_grid[y, x][key] = val
                        continue
                    if key not in matching_dict.keys():
                        continue
                    new_id = matching_dict[key]
                    if new_id not in new_grid[y, x].keys():
                        new_grid[y, x][new_id] = val
                    else:
                        prev = new_grid[y, x][new_id]
                        prev_h = prev[1] if len(prev) >= 2 else 0.0
                        val_h = val[1] if len(val) >= 2 else 0.0
                        prev_c = prev[2] if len(prev) >= 3 else 0
                        val_c = val[2] if len(val) >= 3 else 0
                        if val_h > prev_h:
                            prev[1] = val_h
                            if len(val) >= 4:
                                if len(prev) >= 4:
                                    prev[3] = val[3]
                                else:
                                    prev.append(val[3])
                        prev[2] = prev_c + val_c

            for i in range(self.gs):
                for j in range(self.gs):
                    if new_grid[i, j] is None:
                        new_grid[i, j] = {}

            self.grid = new_grid.copy()
            self.instance_dict = new_instance_dict.copy()
            init = False
            if not updated:
                break

        if self.rot_map:
            for id, val in self.instance_dict.items():
                if self.hm3dsem_mat_mode:
                    self.instance_dict[id]["mask"] = np.rot90(val["mask"], k=1)
                else:
                    self.instance_dict[id]["mask"] = val["mask"].T
