import os, sys
import json
import argparse
import torch

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
    parser.add_argument("--max-depth", type=int, default=3,
                        help="Maximum depth value [m] (Default: 3)")

    
    args = parser.parse_args()
    parser.add_argument("--version", type=str, default=args.vlm,
                        help="Version name to append to the output map name (e.g., grid_lseg_v1.npy)")




    args, remaining_args = parser.parse_known_args()
    if args.vlm == "seem":
        parser.add_argument("--feat-dim", type=int, default=512,
                            help="Dimension of the SEEM feature vector (Default: 512)")
        parser.add_argument("--threshold-confidence", type=float, default=0.5,
                            help="Threshold of confidence score for SEEM (Default: 0.5)")
    elif args.vlm == "lseg":
        parser.add_argument('--lseg-ckpt', type=str, default='/home/hong/VLMAPS/vlmaps_oop/lseg/ckpt/demo_e200.ckpt')
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
    NotImplementedError