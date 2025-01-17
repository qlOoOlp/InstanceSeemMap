import os, sys
import json
import argparse
import torch
from map.utils.matterport3d_categories import mp3dcat

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu"], help="Select device to use (Default: cuda)")
    parser.add_argument("--vlm", default="seem", type = str, choices=["lseg","seem"],
                        help="Name of the vln model to use (Default: seem)")
    parser.add_argument("--data-type", type=str, default="habitat_sim",
                        choices=["habitat_sim", "rtabmap"], help="Select data type to use (Default: habitat_sim)")
    parser.add_argument("--scene-id", type=str, default="2t7WUuJeko7_2",
                        help="Scene name to use (Default: 2t7WUuJeko7_2)")
    parser.add_argument("--only-gt", action = "store_true",
                        help="Only make ground truth map")

    # data path
    now_root = os.getcwd()
    now_root = os.path.join(now_root, "Data")
    parser.add_argument("--root-path", default = now_root, type=str,
                        help="Root path to use")
    

    # map scale
    parser.add_argument("--cs", type=float, default=0.025,
                        help="Grid map cell size [m] (Default: 0.025)")
    parser.add_argument("--gs", type=int, default=2000,
                        help="Number of cells per side of the grid map (Default: 2000)")
    parser.add_argument("--depth-sample-rate", type=int, default=10,
                        help="Depth image sampling rate (Default: 10)")
    parser.add_argument("--camera-height", type=float, default=1.5,
                        help="Camera height from the ground [m] (Default: 1.5)")
    parser.add_argument("--max-depth", type=float, default=3,
                        help="Maximum depth value [m] (Default: 3)")
    parser.add_argument("--min-depth", type=float, default=1,
                        help="Minimum depth value [m] (Default: 0.1)")
    parser.add_argument("--start-frame", type=int, default=0,)
    parser.add_argument("--end-frame", type=int, default=-1)

    
    # args = parser.parse_args()


    args, remaining_args = parser.parse_known_args()
    parser.add_argument("--version", type=str, default=args.vlm if not args.only_gt else "gt",
                        help="Version name to append to the output map name (e.g., grid_lseg_v1.npy)")
    if args.data_type == "habitat_sim":
        parser.add_argument("--dataset-type", type=str, default="mp3d",
                        choices=["mp3d","replica","scannet"],help="Dataset type to use (Default: mp3d)")
    if args.vlm == "seem":
        parser.add_argument("--feat-dim", type=int, default=512,
                            help="Dimension of the SEEM feature vector (Default: 512)")
        parser.add_argument("--threshold-confidence", type=float, default=0.9,
                            help="Threshold of confidence score for SEEM (Default: 0.5)")
        parser.add_argument("--seem-type", type=str, default="base",
                            choices=["base","obstacle","tracking","bbox","dbscan","floodfill"],
                            help="Type of SEEM to use (Default: base)")
        parser.add_argument("--downsampling-ratio",type=float, default=1,
                            help="Downsampling ratio for SEEM input RGB image (Default: 1)")
        args, remaining_args = parser.parse_known_args(remaining_args)
        if args.seem_type != "base":
            parser.add_argument("--no-submap", action="store_false",
                                help="Do not make rgb map")
            parser.add_argument("--using-seemID", action="store_true",
                                help = "Use SEEM category ID for instance ID")
            parser.add_argument("--upsample", action="store_true",
                                help="Upsample the SEEM feature map before using it")
            parser.add_argument("--no-IQR", action="store_false",
                                help="Apply IQR-based preprocessing to remove outlier depth values for each instance")
            parser.add_argument("--min-size-denoising-after-projection", type=int, default=5,
                                help="Minimum size of instance after denoising projected features to keep it (Default: 5)")
            parser.add_argument("--threshold-pixelSize", type=int, default=25,
                                help="Threshold of pixel size for SEEM feature (Default: 25)")
            parser.add_argument("--threshold-semSim", type=float, default=0.85,
                                help="Threshold of semantic similarity for SEEM feature (Default: 0.85)")
            parser.add_argument("--threshold-geoSim", type=float, default=0.4,
                                help="Threshold of geometric similarity for SEEM feature (Default: 0.4)")
            parser.add_argument("--threshold-bbox", type=float, default=0.6,
                                help="Threshold of bbox iou (Default: 0.4)")
            parser.add_argument("--threshold-semSim-post", type=float, default=0.85,
                                help="Threshold of semantic similarity for SEEM feature (Default: 0.85)")
            parser.add_argument("--threshold-geoSim-post", type=float, default=0.4,
                                help="Threshold of geometric similarity for SEEM feature (Default: 0.4)")
            parser.add_argument("--threshold-pixelSize-post", type=int, default=50,
                                help="Threshold of pixel size for SEEM feature (Default: 100)")
            parser.add_argument("--no-postprocessing", action="store_false",
                                help="Do not apply postprocessing to the SEEM feature map")
            parser.add_argument("--max-height", type=float, default=0.5,
                                help="Maximum height of the instance [m] (Default: 0.5)")
            parser.add_argument("--not-using-size", action="store_false",
                                help="Use size information for SEEM feature")
    elif args.vlm == "lseg":
        parser.add_argument('--lseg-ckpt', type=str, default=os.path.join(os.getcwd(),"map/lseg/ckpt/demo_e200.ckpt"))
        parser.add_argument('--crop-size', type=int, default=480)
        parser.add_argument('--base-size', type=int, default=520)
        parser.add_argument('--lang', type=str, default='door,chair,ground,ceiling,other')
        parser.add_argument('--clip-version', type=str, default='ViT-B/32', choices=['ViT-B/16', 'ViT-B/32', 'RN101'])


    if args.data_type == "habitat_sim":
        pass
    elif args.data_type == "rtabmap":
        pass
    else:
        raise ValueError(f"Invalid data type: {args.data_type}")
    

    args = parser.parse_args()
    print(args)
    if args.data_type=='habitat_sim':
        save_args(args)
    return args



def save_args(args):
    if args.data_type == "habitat_sim":
        data_path = os.path.join(args.root_path, args.data_type, args.dataset_type)
    else:
        data_path = os.path.join(args.root_path, args.data_type)
    args.img_save_dir = os.path.join(data_path, args.scene_id)
    if not os.path.exists(args.img_save_dir):
        FileNotFoundError(f"Invalid scene ID: {args.scene_id}")
    param_save_dir = os.path.join(args.img_save_dir, 'map')
    param_save_dir = os.path.join(param_save_dir, f'{args.scene_id}_{args.version}')
    print(param_save_dir)
    os.makedirs(param_save_dir, exist_ok=True)
    # if not os.path.exists(param_save_dir):
    #     os.mkdir(param_save_dir)
    # sub_save_dir = max([0]+[int(e) for e in os.listdir(param_save_dir)])+1
    # param_sub_save_dir = os.path.join(param_save_dir, str(sub_save_dir))
    # os.makedirs(param_sub_save_dir)
    with open(os.path.join(param_save_dir, 'hparam.json'), 'w') as f:
        write_args = args.__dict__.copy()
        del write_args['device']
        json.dump(write_args, f, indent=4)
    

def parse_args_load_map():
    raise NotImplementedError


def parse_args_indexing_map():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu"], help="Select device to use (Default: cuda)")
    # parser.add_argument("--vlm", default="seem", type = str, choices=["lseg","seem"],
    #                     help="Name of the vln model to use (Default: seem)")
    parser.add_argument("--data-type", type=str, default="habitat_sim",
                        choices=["habitat_sim", "rtabmap"], help="Select data type to use (Default: habitat_sim)")
    parser.add_argument("--dataset-type", type=str,default="mp3d",
                        choices=["mp3d","replica","scannet"],help="Dataset type to use (Default: mp3d)")
    parser.add_argument("--scene-id", type=str, default="2t7WUuJeko7_2",
                        help="Scene name to use (Default: 2t7WUuJeko7_2)")
    parser.add_argument("--version", type=str, default="seem",
                        help="Version name to append to the output map name (e.g., grid_lseg_v1.npy)")
    # parser.add_argument("--visualize", action="store_true",
    #                     help="Visualize the map")
    # parser.add_argument("--save-instance-map", action="store_true",
    #                     help="Save the instance map")
    # parser.add_argument("--save-category-map", action="store_true",
    #                     help="Save the category map")
    # parser.add_argument("--indexing-method", type=str, default="height", choices=["height","count","mode"],
    #                     help="Indexing method to use (Default: height)")
    # parser.add_argument("--seem-instance-method", type=str,default="floodfill",choices=["dbscan","floodfill"],
    #                     help="Instance divide method to use for SEEM (Default: floodfill)")
    # parser.add_argument("--query", nargs="+", type=str, default= mp3dcat,
    #                     help="A list of items (space-separated)")
    # parser.add_argument("--threshold-semSim", type=float, default=0.99,
    #                     help="Threshold of semantic similarity for SEEM feature (Default: 0.85)")
    # data path
    now_root = os.getcwd()
    now_root = os.path.join(now_root, "Data")
    parser.add_argument("--root-path", default = now_root, type=str,
                        help="Root path to use")
    args = parser.parse_args()
    print(args)
    return args