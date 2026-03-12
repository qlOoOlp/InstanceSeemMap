import os
import sys
import json
import time
from datetime import datetime, timezone
from map.mapbuilder.utils.mapbuilder import MapBuilder
from utils.parser import parse_args
from omegaconf import OmegaConf


def _resolve_hparam_path(args):
    if args.data_type == "habitat_sim":
        scene_dir = os.path.join(args.root_path, args.data_type, args.dataset_type, args.scene_id)
    else:
        scene_dir = os.path.join(args.root_path, args.data_type, args.scene_id)
    return os.path.join(scene_dir, "map", f"{args.scene_id}_{args.version}", "01buildFeatMap", "hparam.json")


def _update_hparam_timing(args, status, started_at_utc, finished_at_utc, elapsed_sec):
    hparam_path = _resolve_hparam_path(args)
    os.makedirs(os.path.dirname(hparam_path), exist_ok=True)

    payload = {}
    if os.path.exists(hparam_path):
        with open(hparam_path, "r") as f:
            payload = json.load(f)
    else:
        payload = vars(args).copy()
        payload.pop("device", None)

    payload["map_build_started_at_utc"] = started_at_utc
    payload["map_build_finished_at_utc"] = finished_at_utc
    payload["map_build_elapsed_sec"] = round(float(elapsed_sec), 3)
    payload["map_build_status"] = status

    with open(hparam_path, "w") as f:
        json.dump(payload, f, indent=4)


def main():
    args = parse_args()
    config = OmegaConf.create(vars(args))
    mapbuilder = MapBuilder(config)

    started_at_utc = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    status = "success"

    try:
        mapbuilder.buildmap()
    except BaseException:
        status = "failed"
        raise
    finally:
        elapsed_sec = time.perf_counter() - t0
        finished_at_utc = datetime.now(timezone.utc).isoformat()
        _update_hparam_timing(args, status, started_at_utc, finished_at_utc, elapsed_sec)


if __name__ == "__main__":
    main()
