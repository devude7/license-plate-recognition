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
# or
pip install -e .
```

2. Ensure your `.env` file is configured with database


## Running the Application

### 1. Start PostgreSQL Database

Start the PostgreSQL container using Docker Compose:

```bash
docker-compose up -d
```

To stop the database:
```bash
docker-compose down
```

To stop and remove all data:
```bash
docker-compose down -v
```

### 2. Run the API Server

Start the FastAPI server:

```bash
uvicorn src.api.main:app --reload
# or
python -m uvicorn src.api.main:app --reload
```

The API will:
- Automatically create the database schema on startup
- Initialize with sample license plates (XX11111, XX22222, XX33333)
- Be available at `http://localhost:8000`

### 3. Test the API

- **Health check**: `http://localhost:8000/health`
- **API docs**: `http://localhost:8000/docs` 

**Test detection endpoint:**
```bash
curl -X POST "http://localhost:8000/detect" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@path/to/your/image.jpg"
```

## Database

The project uses **psycopg** (psycopg3), which is the official, modern PostgreSQL adapter for Python. 

The database schema is automatically created on API startup via the `ensure_schema()` function.
