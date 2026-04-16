from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")


def execute_notebook(notebook_path: str | Path, input_path: str | Path, work_dir: str | Path) -> dict[str, Any]:
    """Execute the notebook code cells that drive the image pipeline.

    The notebook is treated as the source of truth for the processing logic.
    The helper skips notebook-only install and plotting cells that are not
    needed for the Flask upload flow.
    """

    notebook_path = Path(notebook_path).resolve()
    input_path = Path(input_path).resolve()
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    outputs_dir = work_dir / "outputs"
    if outputs_dir.exists():
        shutil.rmtree(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "stages").mkdir(parents=True, exist_ok=True)

    previous_cwd = Path.cwd()
    previous_env = os.environ.get("IMAGE_INPUT_PATH")
    os.environ["IMAGE_INPUT_PATH"] = str(input_path)

    namespace: dict[str, Any] = {"__name__": "__main__"}

    try:
        os.chdir(work_dir)
        notebook_data = json.loads(notebook_path.read_text(encoding="utf-8"))

        for cell in notebook_data.get("cells", []):
            if cell.get("cell_type") != "code":
                continue

            source = "".join(cell.get("source", []))
            stripped = source.lstrip()
            if not stripped:
                continue
            if stripped.startswith("!") or stripped.startswith("%"):
                continue
            if "from google.colab import files" in source:
                continue
            if "# Select ONE image" in source:
                break

            exec(compile(source, str(notebook_path), "exec"), namespace)

    finally:
        os.chdir(previous_cwd)
        if previous_env is None:
            os.environ.pop("IMAGE_INPUT_PATH", None)
        else:
            os.environ["IMAGE_INPUT_PATH"] = previous_env

    return namespace
