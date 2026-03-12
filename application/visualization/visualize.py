import os
import traceback

from utils.utils import load_config
from .visualizer import visualizer

from tqdm import tqdm


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _iter_jobs(config):
    """
    Expand hierarchical visualize config to flat single-run configs.
    Backward compatible with the legacy flat config.
    """
    runs = config.get("runs")
    if not runs:
        yield config
        return

    base = {k: v for k, v in config.items() if k not in ("common", "runs")}
    common = config.get("common", {})
    base.update(common)

    for run in runs:
        run_base = dict(base)
        run_base.update({k: v for k, v in run.items() if k != "datasets"})

        datasets = _as_list(run.get("datasets"))
        if not datasets:
            if "scene_id" in run_base:
                yield run_base
            continue

        for dataset in datasets:
            ds_base = dict(run_base)
            ds_base.update(
                {k: v for k, v in dataset.items() if k not in ("scenes", "scene_ids")}
            )

            scenes = dataset.get("scenes")
            scene_ids = dataset.get("scene_ids")

            if scenes is not None:
                for scene in _as_list(scenes):
                    scene_cfg = {"scene_id": scene} if isinstance(scene, str) else dict(scene)
                    job = dict(ds_base)
                    job.update(scene_cfg)
                    yield job
                continue

            if scene_ids is not None:
                for scene_id in _as_list(scene_ids):
                    job = dict(ds_base)
                    job["scene_id"] = scene_id
                    yield job
                continue

            if "scene_id" in ds_base:
                yield ds_base


def _resolve_output_path(job):
    if not job.get("save_image", True):
        return job.get("output_path")

    explicit = job.get("output_path")
    if explicit:
        return explicit

    output_root = job.get("output_root")
    if output_root:
        scene_tag = f"{job['scene_id']}_{job['version']}"
        return os.path.join(output_root, job["vlm"], job["dataset_type"], scene_tag)

    # None -> visualizer default: <target_dir>/viz
    return None


def _validate_job(job):
    required = [
        "data_path",
        "data_type",
        "dataset_type",
        "scene_id",
        "version",
        "gt_version",
        "vlm",
        "visualize",
    ]
    missing = [k for k in required if k not in job]
    return missing


def _log(msg, pbar=None):
    if pbar is not None:
        pbar.write(msg)
    else:
        print(msg)


def __main__():
    config = load_config("config/visualize.yaml")
    if config is None:
        raise FileNotFoundError("Failed to load config/visualize.yaml")

    jobs = list(_iter_jobs(config))
    if not jobs:
        raise ValueError("No visualization job found in config.")

    total = len(jobs)
    _log(f"[visualize] total jobs: {total}")

    pbar = tqdm(total=total, desc="Visualizing scenes", unit="scene") if tqdm else None
    ok = 0
    fail = 0
    skip = 0

    for idx, job in enumerate(jobs, start=1):
        missing = _validate_job(job)
        if missing:
            _log(
                f"[visualize] [{idx}/{total}] skip job due to missing keys: {missing}. "
                f"job={job}",
                pbar,
            )
            skip += 1
            if pbar is not None:
                done = ok + fail + skip
                pbar.update(1)
                pbar.set_postfix(done=f"{done}/{total}", ok=ok, fail=fail, skip=skip)
            continue

        output_path = _resolve_output_path(job)
        tag = (
            f"vlm={job['vlm']}, dataset={job['dataset_type']}, "
            f"scene={job['scene_id']}, version={job['version']}"
        )
        _log(f"[visualize] [{idx}/{total}] start {tag}", pbar)

        try:
            viz = visualizer(job, output_path=output_path)
            viz.visualize()
            ok += 1
            _log(f"[visualize] [{idx}/{total}] done {tag}", pbar)
        except Exception as exc:
            fail += 1
            _log(f"[visualize] [{idx}/{total}] failed {tag}: {exc}", pbar)
            traceback.print_exc()
        finally:
            if pbar is not None:
                done = ok + fail + skip
                pbar.update(1)
                pbar.set_postfix(done=f"{done}/{total}", ok=ok, fail=fail, skip=skip)

    if pbar is not None:
        pbar.close()
    _log(f"[visualize] finished. total={total}, ok={ok}, fail={fail}, skip={skip}")


if __name__ == "__main__":
    __main__()
