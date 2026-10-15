import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NOTEBOOKS = {
    "laya": {
        "dir": ROOT / "kaggle" / "laya",
        "notebook": "laya.ipynb",
        "scratch": "/tmp/s6e10-laya",
        "submission": "submission_laya.csv",
        "epoch": 'SCRATCH / f"laya-epoch-{epoch}"',
        "kernel_id": "YOUR_KAGGLE_USERNAME/s6e10-laya",
        "title": "S6E10 Laya",
    },
    "kev": {
        "dir": ROOT / "kaggle" / "kev-0.8b",
        "notebook": "kev-0.8b.ipynb",
        "scratch": "/tmp/s6e10-kev",
        "submission": "submission_kev.csv",
        "epoch": 'SCRATCH / f"kev-epoch-{epoch}"',
        "kernel_id": "YOUR_KAGGLE_USERNAME/s6e10-kev-0.8b",
        "title": "S6E10 Kev-0.8B",
    },
}


def notebook_text(spec):
    path = spec["dir"] / spec["notebook"]
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = ["".join(cell["source"]) for cell in notebook["cells"]]
    return notebook, cells


def test_local_notebooks_unchanged():
    result = subprocess.run(
        ["git", "diff", "--exit-code", "--", "notebooks"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_kaggle_notebooks_use_kaggle_contract():
    for spec in NOTEBOOKS.values():
        notebook, cells = notebook_text(spec)
        assert len(cells) == 6
        joined = "\n".join(cells)
        assert "imported_from" not in joined
        assert "--no-deps" in cells[1]
        assert "torch_before != torch_after" in cells[1]
        assert "/kaggle/input/playground-series-s6e10" in cells[2]
        assert f'SCRATCH = Path("{spec["scratch"]}")' in cells[2]
        assert "dataset/playground-series-s6e10" not in cells[2]
        assert "repo_root" not in cells[2]
        assert f'WORK / "{spec["submission"]}"' in cells[5]
        assert spec["epoch"] in cells[5]
        assert "Path.cwd()" not in cells[5]


def test_kernel_metadata():
    for spec in NOTEBOOKS.values():
        metadata = json.loads((spec["dir"] / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert metadata["id"] == spec["kernel_id"]
        assert metadata["title"] == spec["title"]
        assert metadata["code_file"] == spec["notebook"]
        assert metadata["language"] == "python"
        assert metadata["kernel_type"] == "notebook"
        assert metadata["is_private"] == "true"
        assert metadata["enable_gpu"] == "true"
        assert metadata["enable_internet"] == "true"
        assert metadata["machine_shape"] == "NvidiaTeslaT4"
        assert metadata["competition_sources"] == ["playground-series-s6e10"]
        assert metadata["dataset_sources"] == []
        assert metadata["kernel_sources"] == []
        assert metadata["model_sources"] == []


def test_readme_points_at_kaggle_notebooks():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "kaggle/laya/laya.ipynb" in readme
    assert "kaggle/kev-0.8b/kev-0.8b.ipynb" in readme
    assert "kaggle kernels push -p kaggle/laya" in readme
    assert "kaggle kernels push -p kaggle/kev-0.8b" in readme
    assert "Create a notebook from `notebooks/laya.ipynb`" not in readme
