"""Run the approved models on Kaggle's GPU for a survey, and bring the result back.

The approved stack (D+ roofs: our U-Net + Mask R-CNN with the teammate's Inria and UAVid
models as extra inputs; land cover v2: SegFormer-B2) needs a GPU, so it does not run on the
laptop. This module does the light part locally and the heavy part on Kaggle:

    1. warp the image to its UTM zone at 0.3 m (the models' scale) and save it as job_<stem>.npz
    2. upload it as a new version of the private dataset <user>/cadastraai-job-inputs
    3. push the kernel <user>/cadastraai-process (training/deploy/process_bundles.py)
    4. wait for it, download the bundles, build the surveys (import_bundle.build_survey)

Needs a Kaggle API token for the account that owns the model datasets. One job at a time.
"""
import json
import shutil
import threading
import time
from pathlib import Path

import numpy as np

import kaggle_dns  # noqa: F401  (resolver fallback for api.kaggle.com on some networks)

ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = ROOT / "data" / "kaggle_jobs"
GSD_M = 0.3
MODEL_DATASETS = ["cadastraai-friend-models", "cadastraai-roofs-v1"]
MODEL_KERNELS = ["cadastraai-friend-bakeoff", "cadastraai-landcover-v2-segformer", "cadastraai-parcel-boundary", "cadastraai-boundary-india"]
SOURCES = {"bakeoff_friend.py": ROOT / "training/ensemble/bakeoff_friend.py",
           "demo_bundles.py": ROOT / "training/deploy/demo_bundles.py",
           "process_bundles.py": ROOT / "training/deploy/process_bundles.py"}
POLL_S = 20
TIMEOUT_S = 45 * 60
_lock = threading.Lock()

MODEL_INFO = {
    "key": "stack_dplus_landcover_v2",
    "name": "Roofs: stacked ensemble (our U-Net + Mask R-CNN, teammate Inria + UAVid maps); "
            "land cover: SegFormer-B2 (OpenEarthMap); run on Kaggle GPU",
    "summary": "Roofs, fair Gandhinagar exam: 88% of houses found, 12.5% of touching pairs merged, outline IoU 0.89. "
               "Roofs the stack misses but land cover marks as building are filled in and tagged for review. "
               "Land cover, OpenEarthMap validation: mIoU 0.67 (road 0.65, tree 0.71, grass 0.58, bare land 0.44). "
               "Parcels: plots grow from each house and stop at the parcel-boundary model's lines (U-Net trained on Dutch "
               "cadastral parcels, fine-tuned on hand-labelled Indian plots): boundary F 0.61 on held-out Indian crops "
               "(0.49 before the India fine-tune); on a held-out Dutch city 53% of official parcels matched at IoU 0.5.",
    "limits": "Roofs trained on one planned Indian sector plus WHU; misses some red-tile / dark roofs and very "
              "dense blocks. Land cover weakest on bare land (IoU 0.44). Everything is processed at 0.3 m.",
}


def api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    a = KaggleApi()
    a.authenticate()
    return a


def available():
    """True when a Kaggle token is configured and the API answers."""
    try:
        a = api()
        a.kernels_status(f"{_user(a)}/cadastraai-demo-bundles")
        return True
    except Exception:
        return False


def _user(a):
    return a.config_values.get("username") or a.get_config_value("username") or "shikkoustic"


def prepare(job_dir, stem, loaded, name, imagery):
    """Save one image for the kernel. loaded: segment.load_survey(..., gsd_m=0.3) output."""
    t = loaded["transform"]
    info = {"name": name, "crs": loaded["crs"].to_string(), "transform": [t.a, t.b, t.c, t.d, t.e, t.f],
            "gsd_m": abs(t.a), "imagery": imagery}
    inputs = job_dir / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(inputs / f"job_{stem}.npz", rgb=loaded["rgb"], valid=loaded["valid"], info=json.dumps(info))


def _wait_dataset(a, ref, stage):
    t0 = time.time()
    while time.time() - t0 < 15 * 60:
        try:
            if str(a.dataset_status(ref)).strip().strip('"').lower() == "ready":
                return
        except Exception:
            pass
        stage("Uploading the image to Kaggle")
        time.sleep(10)
    raise RuntimeError("Kaggle did not finish preparing the uploaded image in time.")


def _launcher():
    files = {k: p.read_text(encoding="utf-8") for k, p in SOURCES.items()}
    return ("import subprocess, sys\nFILES = " + repr(files) + "\n"
            "for name, src in FILES.items():\n    open('/kaggle/working/' + name, 'w').write(src)\n"
            "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'segmentation-models-pytorch==0.5.0', 'rasterio'], check=True)\n"
            "subprocess.run([sys.executable, '/kaggle/working/process_bundles.py'], check=True)\n")


def run(job_dir, stage=lambda s: None):
    """Upload job_dir/inputs, run the kernel, download bundles to job_dir/out. Returns bundle paths."""
    with _lock:
        a = api()
        user = _user(a)
        ds_ref, k_ref = f"{user}/cadastraai-job-inputs", f"{user}/cadastraai-process"
        inputs = job_dir / "inputs"
        (inputs / "dataset-metadata.json").write_text(json.dumps(
            {"title": "cadastraai-job-inputs", "id": ds_ref, "licenses": [{"name": "CC0-1.0"}]}))
        stage("Uploading the image to Kaggle")
        exists = True
        try:
            a.dataset_status(ds_ref)
        except Exception as e:          # Kaggle answers 403 (not 404) for a private dataset that does not exist yet
            if not any(c in str(e) for c in ("403", "404", "Not Found")):
                raise
            exists = False
        if exists:
            a.dataset_create_version(str(inputs), version_notes=job_dir.name, quiet=True, delete_old_versions=True, dir_mode="zip")
        else:
            a.dataset_create_new(str(inputs), public=False, quiet=True, dir_mode="zip")
        time.sleep(15)
        _wait_dataset(a, ds_ref, stage)

        kdir = job_dir / "kernel"
        kdir.mkdir(exist_ok=True)
        (kdir / "run.py").write_text(_launcher(), encoding="utf-8")
        (kdir / "kernel-metadata.json").write_text(json.dumps({
            "id": k_ref, "title": "cadastraai-process", "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [f"{user}/{d}" for d in MODEL_DATASETS] + [ds_ref],
            "kernel_sources": [f"{user}/{k}" for k in MODEL_KERNELS], "competition_sources": []}))
        stage("Running the approved models on Kaggle GPU")
        a.kernels_push(str(kdir))
        time.sleep(30)
        t0 = time.time()
        while True:
            res = a.kernels_status(k_ref)
            status = getattr(res, "status", None) or json.loads(str(res)).get("status")
            status = str(status).upper()
            if "COMPLETE" in status:
                break
            if "ERROR" in status or "CANCEL" in status:
                raise RuntimeError(f"Kaggle run failed ({status}); see the log of {k_ref}.")
            if time.time() - t0 > TIMEOUT_S:
                raise RuntimeError("Kaggle run took too long.")
            time.sleep(POLL_S)
        stage("Downloading the results")
        out = job_dir / "out"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir()
        a.kernels_output(k_ref, path=str(out), quiet=True)
        return sorted(out.rglob("bundles/*.npz"))


def new_job_dir(tag):
    d = JOBS_DIR / (time.strftime("%Y%m%d-%H%M%S") + "-" + tag)
    d.mkdir(parents=True, exist_ok=True)
    return d
