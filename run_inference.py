# Bowling Segmentation - Video Inference Script
# Runs the trained Mask R-CNN (ball, lane, pins) on bowling videos.
# Predicts HIT or MISS based on ball trajectory relative to lane and pins.
#
# Usage:
#     Set VIDEO_PATH to a file path for recorded video
#     Set VIDEO_PATH to "live" for webcam live feed
#     Press 'q' to quit live mode

import os
import sys
import cv2
import numpy as np
import torch
from torchvision import transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import time

# BGR colors for OpenCV
LANE_COLOR = (0, 165, 255)       # orange
PIN_COLOR = (0, 255, 255)        # yellow
HIT_COLOR = (0, 255, 0)          # green
MISS_COLOR = (0, 0, 255)         # red
PENDING_COLOR = (255, 255, 255)  # white

# ===== INSERT YOUR VIDEO PATH HERE (or "live" for webcam) =====
VIDEO_PATH = r"C:\Users\okmar\Desktop\bowling\missed\0425(6).mp4"
# VIDEO_PATH = "live"
# ===============================================================


def get_model(num_classes, weights_path, device):
    model = maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def draw_detection(overlay, mask_np, box, score, label, color):
    """Draw mask overlay, bounding box, and label for a single detection."""
    mask = mask_np > 0.5
    color_arr = np.array(color, dtype=np.uint8)
    overlay[mask] = (overlay[mask] * 0.5 + color_arr * 0.5).astype(np.uint8)
    ibox = box.astype(int)
    cv2.rectangle(overlay, (ibox[0], ibox[1]), (ibox[2], ibox[3]), color, 2)
    cv2.putText(overlay, f"{label}: {score:.2f}", (ibox[0], ibox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def process_frame(model, frame, device, transform, score_threshold, prediction, ball_was_on_lane):
    """Process a single frame and return the overlay, updated prediction, and ball_was_on_lane."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = transform(rgb).to(device)

    with torch.no_grad():
        preds = model([img_tensor])[0]

    overlay = frame.copy()
    masks = preds['masks']
    scores = preds['scores']
    boxes = preds['boxes']
    labels = preds['labels']

    balls = []
    lane_dets = []
    pin_dets = []

    for i in range(len(scores)):
        if scores[i] < score_threshold:
            continue
        label_id = labels[i].item()
        box = boxes[i].cpu().numpy()
        score_val = scores[i].item()
        mask_np = masks[i, 0].cpu().numpy()
        if label_id == 1:
            balls.append((box, score_val, mask_np))
        elif label_id == 2:
            lane_dets.append((box, score_val, mask_np))
        elif label_id == 3:
            pin_dets.append((box, score_val, mask_np))

    best_ball = max(balls, key=lambda x: x[1]) if balls else None
    best_lane = max(lane_dets, key=lambda x: x[1]) if lane_dets else None

    # --- Hit/Miss prediction (only update while PENDING) ---
    if prediction == "PENDING" and best_ball and best_lane:
        bb = best_ball[0]
        lb = best_lane[0]

        ball_cx = (bb[0] + bb[2]) / 2
        ball_cy = (bb[1] + bb[3]) / 2
        ball_diameter = max(bb[2] - bb[0], bb[3] - bb[1])
        ball_radius = ball_diameter / 2

        lane_mid_y = (lb[1] + lb[3]) / 2
        ball_on_lane = (lb[0] <= ball_cx <= lb[2])

        if ball_on_lane:
            ball_was_on_lane = True

        if ball_was_on_lane and not ball_on_lane:
            prediction = "MISS"
        elif ball_on_lane:
            ball_past_middle = False
            if pin_dets:
                pin_avg_y = sum((p[0][1] + p[0][3]) / 2 for p in pin_dets) / len(pin_dets)
                if pin_avg_y < lane_mid_y:
                    ball_past_middle = ball_cy < lane_mid_y
                else:
                    ball_past_middle = ball_cy > lane_mid_y

            if ball_past_middle and pin_dets:
                pin_cx_avg = sum((p[0][0] + p[0][2]) / 2 for p in pin_dets) / len(pin_dets)
                diff = abs(ball_cx - pin_cx_avg)
                if diff < ball_radius:
                    prediction = "HIT"
                else:
                    prediction = "MISS"

    # --- Draw detections ---
    for box, score_val, mask_np in lane_dets:
        draw_detection(overlay, mask_np, box, score_val, "lane", LANE_COLOR)

    for box, score_val, mask_np in pin_dets:
        draw_detection(overlay, mask_np, box, score_val, "pins", PIN_COLOR)

    if prediction == "HIT":
        ball_color = HIT_COLOR
    elif prediction == "MISS":
        ball_color = MISS_COLOR
    else:
        ball_color = PENDING_COLOR

    for box, score_val, mask_np in balls:
        draw_detection(overlay, mask_np, box, score_val, "ball", ball_color)

    # Draw prediction banner
    if prediction != "PENDING":
        pred_color = HIT_COLOR if prediction == "HIT" else MISS_COLOR
        cv2.rectangle(overlay, (10, 10), (420, 70), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 10), (420, 70), pred_color, 2)
        cv2.putText(overlay, f"PREDICTION: {prediction}", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, pred_color, 3)

    return overlay, prediction, ball_was_on_lane


def run_live(model, device, score_threshold=0.5):
    """Run inference on live webcam feed. Press 'q' to quit, 'r' to reset prediction."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        sys.exit(1)

    print("Live webcam mode - Press 'q' to quit, 'r' to reset prediction")
    transform = T.ToTensor()
    prediction = "PENDING"
    ball_was_on_lane = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        overlay, prediction, ball_was_on_lane = process_frame(
            model, frame, device, transform, score_threshold, prediction, ball_was_on_lane
        )

        cv2.imshow("Bowling Inference - Live", overlay)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            prediction = "PENDING"
            ball_was_on_lane = False
            print("Prediction reset.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinal Prediction: {prediction}")


def run_video(model, video_path, output_path, device, score_threshold=0.5):
    """Run inference on a recorded video file with live display. Press 'q' to quit."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return False, "UNKNOWN"

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    delay = max(1, int(1000 / fps))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    transform = T.ToTensor()
    prediction = "PENDING"
    ball_was_on_lane = False
    frame_idx = 0

    print("Displaying video - Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        overlay, prediction, ball_was_on_lane = process_frame(
            model, frame, device, transform, score_threshold, prediction, ball_was_on_lane
        )

        out.write(overlay)
        frame_idx += 1

        # Show frame-by-frame in a window
        cv2.imshow("Bowling Inference", overlay)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            print("Stopped early by user.")
            break

        if frame_idx % 30 == 0:
            print(f"    Frame {frame_idx}/{total_frames}", end='\r')

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"    {frame_idx} frames | Prediction: {prediction}        ")
    return True, prediction


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    WEIGHTS = os.path.join(BASE_DIR, "bowling_maskrcnn_3class.pth")
    OUTPUT_DIR = os.path.join(BASE_DIR, "inference_output")
    SCORE_THRESHOLD = 0.5

    if not os.path.exists(WEIGHTS):
        print(f"ERROR: Model weights not found at {WEIGHTS}")
        print("Run train_segmentation.py first!")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model from {WEIGHTS}...")
    model = get_model(num_classes=4, weights_path=WEIGHTS, device=device)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    video_path = VIDEO_PATH

    # --- Live webcam mode ---
    if video_path.lower() == "live":
        print("\nMode: LIVE WEBCAM")
        run_live(model, device, SCORE_THRESHOLD)
        return

    # --- Recorded video mode ---
    if not os.path.exists(video_path):
        print(f"ERROR: Video not found: {video_path}")
        print("Please update VIDEO_PATH at the top of this script.")
        sys.exit(1)

    print(f"\nMode: RECORDED VIDEO")
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"seg_{base_name}.mp4")

    print(f"Running inference on: {video_path}")
    t0 = time.time()
    success, prediction = run_video(model, video_path, out_path, device, SCORE_THRESHOLD)
    elapsed = time.time() - t0

    if not success:
        sys.exit(1)

    print(f"\nPrediction: {prediction}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Output saved to: {out_path}")


if __name__ == "__main__":
    main()
