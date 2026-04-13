from __future__ import annotations

import json
import shutil
import uuid
from collections import defaultdict
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from notebook_runner import execute_notebook

BASE_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = BASE_DIR / "Image_enhancement_algorithm_combining_histogram_equalization_and__bilateral_filtering.ipynb"
HTML_PATH = BASE_DIR / "flask-dashboard.html"
JOB_ROOT = BASE_DIR / "jobs"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

METHOD_ORDER = ["Original", "HE", "CLAHE", "RD", "ESIHE", "RGHS", "Fusion", "Our"]
STAGE_ORDER = ["Original", "HE", "Bilateral", "Wavelet Fusion"]


def _file_url(job_id: str, relative_path: str) -> str:
    return f"/jobs/{job_id}/outputs/{relative_path}"


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, HTML_PATH.name)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/process")
def process_upload():
    uploaded_files = request.files.getlist("files") or request.files.getlist("file")
    if not uploaded_files:
        return jsonify({"error": "No files uploaded."}), 400

    job_id = uuid.uuid4().hex
    job_dir = JOB_ROOT / job_id
    input_dir = job_dir / "input"
    output_dir = job_dir / "outputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for uploaded in uploaded_files:
        if not uploaded.filename:
            continue
        target = input_dir / Path(uploaded.filename).name
        uploaded.save(target)
        saved_paths.append(target)

    zip_files = [path for path in saved_paths if path.suffix.lower() == ".zip"]
    notebook_input = zip_files[0] if zip_files else input_dir

    namespace = execute_notebook(NOTEBOOK_PATH, notebook_input, job_dir)
    results = namespace.get("results", [])
    stage_results = namespace.get("stage_results", [])

    by_image: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        by_image[row["Image"]].append(row)

    stage_by_image: dict[str, list[dict]] = defaultdict(list)
    for row in stage_results:
        stage_by_image[row["Image"]].append(row)

    images = []
    for image_name, rows in by_image.items():
        method_lookup = {row["Method"]: row for row in rows}
        methods = []
        for method_name in METHOD_ORDER:
            row = method_lookup.get(method_name)
            if not row:
                continue
            methods.append(
                {
                    "name": method_name,
                    "imageUrl": _file_url(job_id, f"{image_name}_{method_name}.png"),
                    "metrics": {
                        "PSNR": _to_float(row.get("PSNR")),
                        "SSIM": _to_float(row.get("SSIM")),
                        "Entropy": _to_float(row.get("Entropy")),
                        "Contrast": _to_float(row.get("Contrast")),
                    },
                }
            )

        stage_lookup = {row["Stage"]: row for row in stage_by_image.get(image_name, [])}
        stages = []
        for stage_name in STAGE_ORDER:
            row = stage_lookup.get(stage_name)
            if not row:
                continue
            stages.append(
                {
                    "name": stage_name,
                    "imageUrl": _file_url(job_id, f"stages/{image_name}_{stage_name}.png"),
                    "metrics": {
                        "PSNR": _to_float(row.get("PSNR")),
                        "SSIM": _to_float(row.get("SSIM")),
                        "Entropy": _to_float(row.get("Entropy")),
                        "Contrast": _to_float(row.get("Contrast")),
                    },
                }
            )

        images.append({"name": image_name, "methods": methods, "stages": stages})

    summary_rows = namespace.get("table")
    summary = []
    if summary_rows is not None:
        try:
            summary_df = summary_rows.reset_index()
            for _, row in summary_df.iterrows():
                summary.append(
                    {
                        "method": str(row["Method"]),
                        "PSNR": _to_float(row.get("PSNR")),
                        "SSIM": _to_float(row.get("SSIM")),
                        "Entropy": _to_float(row.get("Entropy")),
                        "Contrast": _to_float(row.get("Contrast")),
                    }
                )
        except Exception:
            summary = []

    return jsonify(
        {
            "jobId": job_id,
            "images": images,
            "summary": summary,
        }
    )


@app.get("/jobs/<job_id>/outputs/<path:filename>")
def job_output(job_id: str, filename: str):
    job_output_dir = JOB_ROOT / job_id / "outputs"
    return send_from_directory(job_output_dir, filename)


@app.get("/jobs/<job_id>/input/<path:filename>")
def job_input(job_id: str, filename: str):
    job_input_dir = JOB_ROOT / job_id / "input"
    return send_from_directory(job_input_dir, filename)


if __name__ == "__main__":
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    app.run(host="127.0.0.1", port=5000, debug=True)
