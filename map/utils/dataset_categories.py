from typing import Any, List, Optional, Sequence, Tuple

from map.utils.hm3dsem_categories import build_obstacle_filter, hm3dsem_cat
from map.utils.matterport3d_categories import mp3dcat
from map.utils.replica_categories import replica_cat


def normalize_dataset_type(dataset_type: Any) -> str:
    token = str(dataset_type).strip().lower()
    if token == "replica":
        return "replica"
    if token == "mp3d":
        return "mp3d"
    if token == "hm3dsem":
        return "hm3dsem"
    return token


def resolve_dataset_categories(dataset_type: Any) -> Tuple[str, List[str], str]:
    dataset_key = normalize_dataset_type(dataset_type)
    if dataset_key == "mp3d":
        return dataset_key, mp3dcat, "matterport3d_categories"
    if dataset_key == "replica":
        return dataset_key, replica_cat, "replica_categories"
    if dataset_key == "hm3dsem":
        return dataset_key, hm3dsem_cat, "hm3dsem_categories"
    raise ValueError(
        f"dataset_type '{dataset_type}' (normalized='{dataset_key}') is not supported. "
        "Supported: mp3d, replica, hm3dsem."
    )


def resolve_dataset_semantics(
    dataset_type: Any,
    default_obstacles: Optional[Sequence[str]] = None,
    use_hm3d_obstacle_filter: bool = False,
) -> Tuple[str, List[str], str, Optional[List[str]], str]:
    dataset_key, categories, category_source = resolve_dataset_categories(dataset_type)

    obstacle_items: Optional[List[str]] = None
    obstacle_source = "none"
    if default_obstacles is not None:
        obstacle_items = list(default_obstacles)
        obstacle_source = "config"
        if use_hm3d_obstacle_filter and dataset_key == "hm3dsem":
            obstacle_items = build_obstacle_filter()(categories)
            obstacle_source = "hm3d_filter"

    return dataset_key, categories, category_source, obstacle_items, obstacle_source
