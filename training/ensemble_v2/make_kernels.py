"""Build one Kaggle script kernel per stage: a launcher that writes common.py and the
stage script next to itself, then runs the stage. Usage: python make_kernels.py"""
import json
from pathlib import Path

HERE = Path(__file__).parent
PREP = "shikkoustic/cadastraai-ens-prep"
GN = "shikkoustic/cadastraai-gandhinagar-mosaic"
KERNELS = {
    "cadastraai-ens-unet": ("train_unet.py", [PREP, GN]),
    "cadastraai-ens-yolo": ("train_yolo.py", [PREP, GN]),
    "cadastraai-ens-maskrcnn": ("train_maskrcnn.py", [PREP, GN]),
    "cadastraai-ens-fuse": ("fuse_eval.py", [PREP, GN, "shikkoustic/cadastraai-ens-unet", "shikkoustic/cadastraai-ens-yolo",
                                              "shikkoustic/cadastraai-ens-maskrcnn", "shikkoustic/cadastraai-fair-comparison"]),
}
for kid, (entry, sources) in KERNELS.items():
    d = HERE / "build" / kid
    d.mkdir(parents=True, exist_ok=True)
    files = {n: (HERE / n).read_text() for n in ("common.py", entry)}
    launcher = (f'"""Kaggle launcher for {entry} (ensemble v2)."""\nimport os, runpy, sys\n'
                f"FILES = {json.dumps(files)}\n"
                "os.chdir('/kaggle/working')\n"
                "for n, src in FILES.items():\n    open(n, 'w').write(src)\n"
                "sys.path.insert(0, '/kaggle/working')\n"
                f"runpy.run_path('{entry}', run_name='__main__')\n")
    (d / f"{kid}.py").write_text(launcher)
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"shikkoustic/{kid}", "title": kid, "code_file": f"{kid}.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": sources,
        "machine_shape": "NvidiaTeslaT4"}, indent=2))
    print("built", kid)
