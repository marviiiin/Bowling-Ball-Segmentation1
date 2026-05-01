# Bowling-Ball-Segmentation1
# Bowling Hit/Miss Prediction Using Instance Segmentation

An automated bowling hit/miss prediction system using Mask R-CNN instance segmentation. The model detects bowling balls, lanes, and pins from video frames and predicts whether a throw results in a **HIT** or **MISS** based on spatial relationships between detected objects.

**Authors:** Marvin Osei-Kuffour, Phong Lu
**Institution:** University of South Florida

---

## Project Structure

```
bowling/
├── train_segmentation.py           # Training script (Mask R-CNN fine-tuning)
├── run_inference.py                # Batch inference on hit/ and missed/ folders
├── run_image_inference.py          # Single video inference (input slot)
├── bowling_maskrcnn_3class.pth     # Trained model weights (~168 MB)
│
├── bowling ball segment.coco-segmentation/
│   └── train/                      # Dataset 1: 125 images, ball only
│
├── Bowling.coco-segmentation (1)/
│   └── train/                      # Dataset 2: 503 images, ball/lane/pins
│
├── hit/                            # 42 ground-truth HIT videos
├── missed/                         # 30 ground-truth MISS videos
│
├── inference_output/               # Annotated output videos
│   ├── hit/
│   └── missed/
```

---

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- OpenCV (`opencv-python`)
- pycocotools
- NumPy

Install dependencies:
```bash
pip install torch torchvision opencv-python pycocotools numpy
```

A CUDA-capable GPU is recommended for training and faster inference.

---

## Usage

### 1. Train the Model

Fine-tunes Mask R-CNN on the combined bowling datasets (628 images, 4 classes: background + ball + lane + pins).

```bash
python train_segmentation.py
```

- Trains for 25 epochs with SGD optimizer
- Saves best model checkpoint as `bowling_maskrcnn_3class.pth`
- Best validation loss: 0.2369 (epoch 8)

### 2. Run Inference on a Single Video

```bash
# With command line argument
python run_image_inference.py path/to/video.mp4

# With file dialog (no argument)
python run_image_inference.py
```

- Opens a file picker if no path is provided
- Runs frame-by-frame segmentation and hit/miss prediction
- Saves annotated video to `inference_output/seg_<name>.mp4`
- Prints prediction result (HIT / MISS) to console

### 3. Run Batch Inference

Processes all videos in `hit/` and `missed/` folders and prints an accuracy summary.

```bash
python run_inference.py
```

---

## How It Works

### Segmentation
The system uses a **Mask R-CNN** model with a ResNet-50 + FPN backbone, fine-tuned to detect three classes:

| Class | Color (in output) |
|-------|-------------------|
| Ball  | White / Green / Red |
| Lane  | Orange |
| Pins  | Yellow |

### Hit/Miss Prediction Logic

The prediction uses a state machine (PENDING → HIT or MISS) based on spatial rules:

1. **Lane Tracking** — Track whether the ball has entered the lane boundaries
2. **Gutter Detection** — If the ball leaves the lane after being on it → **MISS**
3. **Direction Check** — Determine which end of the lane the pins are on and whether the ball has passed the midpoint
4. **Alignment Check** — If the ball is past the midpoint and aligned with the pins (horizontal distance < ball radius) → **HIT**, otherwise → **MISS**

Once a prediction is made, it is final for the rest of the video. The ball bounding box turns **green** (HIT) or **red** (MISS), and a prediction banner appears on the frame.

---

## Results

Evaluated on 72 bowling videos:

| Metric | Value |
|--------|-------|
| Overall Accuracy | 43.1% (31/72) |
| Hit Recall | 23.8% (10/42) |
| Miss Recall | 70.0% (21/30) |


