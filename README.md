# Chess Vision Eval

A robust computer vision module designed for a chess-playing robot. This project uses a live camera feed (typically placed at an oblique angle, ~60 degrees) to detect a chessboard, correct its perspective, and accurately identify the positions and types of all chess pieces using a YOLOv8 neural network. 

## Features

- **Perspective Correction:** Automatically detects the 64 squares of a chessboard or falls back to detecting a specific red border. It calculates a homography matrix to map the oblique camera view to a perfect top-down 2D grid.
- **YOLOv8 Piece Detection:** Utilizes a trained YOLOv8 model (`best.pt`) to detect pieces (White/Black, King/Queen/Rook/Bishop/Knight/Pawn) and maps their bounding boxes precisely to the board squares.
- **Fallback Occupancy Detection:** If the YOLO model is missing or fails, the system can use a frame-differencing algorithm (comparing the current board state against an empty board reference) to detect occupied squares and estimate piece color based on brightness.
- **Interactive HUD & Diagnostics:** Provides real-time visual feedback, including a top-down diagram generator, debug layers showing YOLO bounding boxes, and an adjustable border margin.
- **Training Pipeline:** Includes scripts to download datasets from NDJSON annotations, prepare YOLO formats, and train/fine-tune the model with heavy augmentations tailored for challenging lighting conditions (e.g., dark pieces on dark squares).

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/vision-aura-chess2.git
   cd vision-aura-chess2
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Required packages include `opencv-python`, `numpy`, and `ultralytics`.*

## Usage

### 1. Live Detection Application

Run the main application to start the camera feed:
```bash
python main.py
```
*(Optional: Pass a camera index if you have multiple cameras, e.g., `python main.py 1`)*

**Controls:**
- `R`: **Calibrate**. Places the system in calibration mode. Ensure the board is empty and fully visible. It will try to detect the board squares or the red outer border and calculate the homography matrix.
- `I`: **Analyze**. Warps the current frame, runs the YOLO detection, and displays a clean 2D chess diagram with the recognized pieces.
- `+` / `-`: **Adjust Border Margin**. Fine-tunes the crop size dynamically if the red border calibration slightly misaligns the 64 internal squares.
- `S`: **Save**. Saves the current original frame, warped frame, debug view, and diagram to the `output/` directory.
- `ESC` / `Q`: **Quit**.

### 2. Dataset Preparation & Training

If you want to improve piece detection accuracy (e.g., if a Pawn is occasionally detected as a Queen), you can retrain the model.

**Download and format the dataset:**
The project includes a script to parse `chess-pieces.ndjson`, download the raw images, and convert the bounding boxes into YOLO format.
```bash
python prepare_dataset.py
```

**Train the YOLO model:**
Once the dataset is ready, run the training script. This script applies custom augmentations optimized for chess pieces under variable lighting.
```bash
python train.py
```
*Options:*
- `--model yolov8m.pt` (Use a medium model instead of small)
- `--epochs 50` (Change number of epochs)
- `--resume` (Resume interrupted training)

After training, the best model will be automatically copied to `models/best.pt`, where `main.py` will pick it up on its next launch.

## Configuration

Adjust variables in `config.py` to match your physical setup:
- `BOARD_SIZE`: The pixel size of the warped top-down image (default: 800).
- `BORDER_MARGIN`: The fraction of the board representing the red border, baked into the transformation matrix (default: `0.065`).
- `RED_LOWER` / `RED_UPPER`: HSV bounds for detecting the red calibration border.
- `CAMERA_INDEX`: Default OpenCV camera index (default: `0`).

## Troubleshooting

- **Pieces misaligned (e.g., Knight jumps to the next square):**
  Ensure the calibration grid perfectly aligns with the physical squares. If using the red border method, adjust the `BORDER_MARGIN` in `config.py` or use the `+`/`-` keys during execution.
- **Incorrect piece classification (e.g., Pawn detected as Queen):**
  This is a model limitation based on lighting/angles. Save frames where this occurs, label them, and retrain the model using the provided scripts.
- **Calibration fails:**
  Ensure the board is well-lit and not heavily obstructed by shadows. If the red border is not being found, you may need to adjust the HSV values in `config.py`.

## License
MIT License.
