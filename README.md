# IT416 CV Project

A Flask-based dashboard for evaluating low-light image enhancement methods from the notebook pipeline in `Image_enhancement_fixed.ipynb`.

The app lets you upload image files or a ZIP dataset, runs the notebook-driven enhancement pipeline, and returns visual/metric comparisons for multiple methods.

## Features

- Upload one or more images, or a ZIP dataset
- Run enhancement methods and stage-wise outputs through notebook execution
- Compare methods:
  - Original
  - HE
  - CLAHE
  - RD
  - ESIHE
  - RGHS
  - Fusion
  - Our
- Compute and show metrics:
  - PSNR
  - SSIM
  - Entropy
  - Contrast
- View the paper PDF from the dashboard (`paper.pdf` or fallback `IT416_Final_Submission.pdf`)

## Project Structure

- `app.py`: Flask server, upload handling, API routes, output serving
- `notebook_runner.py`: Executes notebook code cells in a controlled runtime for each job
- `Image_enhancement_fixed.ipynb`: Core enhancement and evaluation pipeline
- `flask-dashboard.html`: Frontend dashboard
- `jobs/` (generated): Per-run inputs and outputs

## Requirements

- Python 3.10+
- pip

Python packages used by the app/pipeline:

- flask
- numpy
- pandas
- matplotlib
- opencv-python
- scikit-image
- pywavelets

## Setup

From the project root:

```bash
python -m venv .venv
```

Activate virtual environment:

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

Windows (cmd):

```cmd
.venv\Scripts\activate.bat
```

Install dependencies:

```bash
pip install flask numpy pandas matplotlib opencv-python scikit-image pywavelets
```

## Run

```bash
python app.py
```

Then open:

- http://127.0.0.1:5000/

## How Processing Works

1. User uploads files from dashboard.
2. Server creates a unique job folder under `jobs/<job_id>/`.
3. `notebook_runner.py` sets `IMAGE_INPUT_PATH` and executes relevant notebook code cells.
4. Notebook outputs (enhanced images + stage images + result tables) are written to `jobs/<job_id>/outputs/`.
5. API response includes image URLs and metrics for frontend rendering.

## API Endpoints

- `GET /` - Dashboard page
- `GET /health` - Health check
- `GET /paper.pdf` - Project paper (preferred: `paper.pdf`, fallback: `IT416_Final_Submission.pdf`)
- `POST /api/process` - Upload and process images
- `GET /jobs/<job_id>/outputs/<filename>` - Serve generated outputs
- `GET /jobs/<job_id>/input/<filename>` - Serve original uploaded files

## Upload Notes

- Accepted input style:
  - One or more image files
  - A ZIP file containing images
- If a ZIP is uploaded, it is used as the notebook input source.
- Max upload size is configured to 200 MB.

## Troubleshooting

- `No files uploaded.`
  - Ensure form uses `multipart/form-data` and includes files.
- Import errors (for OpenCV/scikit-image/pywavelets)
  - Re-activate your virtual environment and reinstall dependencies.
- `Paper PDF not found.`
  - Ensure either `paper.pdf` or `IT416_Final_Submission.pdf` exists in project root.

## Notes

- The notebook remains the source of truth for enhancement logic.
- Flask invokes notebook code dynamically instead of duplicating algorithm code in the backend.
