# license-plate-recognition

## Detector

The detector is a **two-stage pipeline**:

1. **Detection** — A custom-trained **YOLO** (Ultralytics) model (`src/models/best.pt`) localizes the license plate in the image and returns a bounding box. The best box (by confidence) is selected and the plate region is cropped.
2. **OCR** — The cropped plate image is passed to **TrOCR** (`microsoft/trocr-base-printed`), a transformer-based vision–language model, which reads the plate text. The output is normalized (uppercase, alphanumeric only) for lookup.

Optional EU-flag trimming is applied to the crop before OCR (configurable via `cut_left_ratio`) to avoid the blue strip affecting reading.

## Dataset & training
The YOLO model was trained on a custom dataset that I co-created, including bounding-box annotations. Training data is prepared from XML annotations and photos via `scripts/prepare_data_for_yolo.py`, which converts them into YOLO format (train/val splits, normalized labels).

- **Dataset**: https://www.kaggle.com/datasets/piotrstefaskiue/poland-vehicle-license-plate-dataset/data

## Setup

### Prerequisites
- Python 3.13+
- Docker and Docker Compose
- UV package manager (recommended) or pip

### Installation

1. Install dependencies:
```bash
uv sync
```

2. Ensure your `.env` file is configured with database


## Running the Application

**Docker Compose** starts Postgres (and optionally the API). From the project root:

```bash
docker compose up -d --build    # start
docker compose down             # stop
docker compose down -v          # stop and remove DB data
```

Choose one of:

| Mode | What to run |
|------|----------------|
| **All in Docker** | `docker compose up -d --build` — API at `http://localhost:8000`|
| **DB in Docker, API on host** | Install dependencies first (see [Installation](#installation)), then `docker compose up -d` (only Postgres) and `uvicorn src.api.main:app --reload` (API at `http://localhost:8000`). |

When running the API locally, install dependencies with `uv sync` before starting the server.

The API creates the schema and sample plates (XX11111, XX22222, XX33333) on startup.

### Test the API

- **Health**: `http://localhost:8000/health`
- **Docs**: `http://localhost:8000/docs`

```bash
curl -X POST "http://localhost:8000/detect" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@path/to/your/image.jpg"
```

## Database

The project uses **psycopg** (psycopg3), which is the official, modern PostgreSQL adapter for Python. 

The database schema is automatically created on API startup via the `ensure_schema()` function.
