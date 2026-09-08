import sys
from PynamicMesh.core.pipelines import run_pipeline
from PynamicMesh.utils.tools import extract_kwargs


def scene_kwargs(scene_cfg):
    """
    Flattens the sections of one scene configuration into run_pipeline keyword arguments.
    The Functional_Map section may contain the advanced options either nested under `fm_params`
    or flat (symmetry_mode, landmark_params, descr_params, fit_params, n_descr, subsample_step,
    refine, dt, verbose); both forms are accepted (they are normalized by run_pipeline).
    """
    scene_cfg = scene_cfg or {}
    fm_cfg = dict(scene_cfg.get("Functional_Map", {}) or {})
    rg_cfg = dict(scene_cfg.get("Reeb_Graph", {}) or {})
    bg_cfg = dict(scene_cfg.get("Basic_Geometry", {}) or {})
    gs_cfg = dict(scene_cfg.get("Graph_similarity", {}) or {})

    fm_params = fm_cfg.pop("fm_params", None)        # keep the nested mapping out of extract_kwargs
    kwargs = extract_kwargs(fm_cfg, rg_cfg, bg_cfg, gs_cfg)
    if fm_params:
        kwargs["fm_params"] = dict(fm_params)
    return kwargs


def run_batch(config, path_str):

    batch_kwargs = {}
    for key, value in config.items():
        if key == "Data":
            continue
        batch_kwargs[key] = scene_kwargs(value)

    if not batch_kwargs:
        print("Error: Batch mode enabled, but no scene configurations found in YAML.", file=sys.stderr)
        sys.exit(1)

    run_pipeline(path_str=path_str, is_batch=True, batch_kwargs=batch_kwargs)