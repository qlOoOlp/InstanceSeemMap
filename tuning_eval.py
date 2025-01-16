import json
import argparse
from application.evaluation.metrics.evaluation import evaluation

def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file", type=str, required=True, help="Path to the JSON output file")
    parser.add_argument("--key", type=str, required=True, help="Key for the result in JSON file")
    parser.add_argument("--min-depth", type=float, required=True, help="Minimum depth")
    parser.add_argument("--min-size-denoising-after-projection", type=int, required=True, help="Minimum size for denoising after projection")
    parser.add_argument("--threshold-pixelSize", type=int, required=True, help="Threshold for pixel size")
    parser.add_argument("--threshold-semSim", type=float, required=True, help="Threshold for semantic similarity")
    parser.add_argument("--threshold-bbox", type=float, required=True, help="Threshold for bounding box")
    parser.add_argument("--threshold-semSim-post", type=float, required=True, help="Threshold for semantic similarity post")
    parser.add_argument("--threshold-geoSim-post", type=float, required=True, help="Threshold for geometric similarity post")
    parser.add_argument("--threshold-pixelSize-post", type=int, required=True, help="Threshold for pixel size post")
    parser.add_argument("--version", type=str, required=True)
    args = parser.parse_args()

    # Evaluation config
    config = {
        "data_path": "/nvme0n1/hong/VLMAPS/InstanceSeemMap/Data",
        "data_type": "habitat_sim",
        "dataset_type": "replica",
        "vlm": "ours",
        "version": args.version,
        "scene_ids": ["apartment_0_1"],
        "gt_version": "gt",
        "visualize": False,
        "bool_save": False,
    }

    # Run evaluation
    eval = evaluation(config)
    eval_results = eval.evaluate()

    # Load existing JSON file
    with open(args.output_file, 'r') as f:
        data = json.load(f)

    # Add new result
    data[args.key] = {
        "params": {
            "min_depth": args.min_depth,
            "min_size_denoising_after_projection": args.min_size_denoising_after_projection,
            "threshold_pixelSize": args.threshold_pixelSize,
            "threshold_semSim": args.threshold_semSim,
            "threshold_bbox": args.threshold_bbox,
            "threshold_semSim_post": args.threshold_semSim_post,
            "threshold_geoSim_post": args.threshold_geoSim_post,
            "threshold_pixelSize_post": args.threshold_pixelSize_post,
        },
        "results": eval_results
    }

    # Save updated JSON file
    with open(args.output_file, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    main()