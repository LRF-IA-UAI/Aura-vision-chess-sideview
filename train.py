"""
Train YOLOv8 on the chess-pieces dataset.

Usage:
    python train.py                  # defaults: yolov8s, 100 epochs
    python train.py --model yolov8m  # medium model (more accurate, slower)
    python train.py --epochs 50      # fewer epochs for quick test
    python train.py --resume         # resume interrupted training

Augmentations are tuned for the black-pieces-on-dark-squares problem:
heavy brightness/contrast jitter so the model learns to distinguish
dark pieces from dark backgrounds under varied lighting.
"""
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Train YOLO chess piece detector")
    parser.add_argument("--model", default="yolov8s.pt",
                        help="Base model: yolov8n.pt, yolov8s.pt, yolov8m.pt (default: yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--device", default=None,
                        help="Device: 0 for GPU, cpu for CPU (auto-detected if omitted)")
    args = parser.parse_args()

    from ultralytics import YOLO

    data_yaml = os.path.join(os.path.dirname(__file__),
                             "datasets", "chess-pieces", "data.yaml")

    if not os.path.exists(data_yaml):
        print(f"ERROR: {data_yaml} not found. Run prepare_dataset.py first.")
        return

    if args.resume:
        last_pt = os.path.join(os.path.dirname(__file__),
                               "runs", "detect", "chess", "weights", "last.pt")
        if not os.path.exists(last_pt):
            print(f"ERROR: No checkpoint found at {last_pt}")
            return
        model = YOLO(last_pt)
        print(f"Resuming from {last_pt}")
    else:
        model = YOLO(args.model)
        print(f"Starting from pretrained {args.model}")

    print(f"Dataset: {data_yaml}")
    print(f"Epochs: {args.epochs}  Batch: {args.batch}  ImgSz: {args.imgsz}")
    print()

    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=os.path.join(os.path.dirname(__file__), "runs", "detect"),
        name="chess",
        exist_ok=True,

        # ── Augmentations for black-on-dark-square robustness ──
        # Heavy brightness & contrast jitter
        hsv_h=0.02,       # hue (small — pieces don't change hue much)
        hsv_s=0.5,        # saturation variation
        hsv_v=0.6,        # value/brightness (aggressive — key for dark pieces)

        # Geometric
        degrees=5.0,       # slight rotation (oblique camera angles)
        translate=0.1,
        scale=0.4,         # scale jitter (pieces at different distances)
        fliplr=0.5,
        mosaic=1.0,        # mosaic augmentation
        mixup=0.15,        # light mixup for generalization

        # Training params
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        cos_lr=True,
        patience=20,       # early stopping
        workers=4,
        verbose=True,
    )

    # Copy best model to models/
    best_src = os.path.join(os.path.dirname(__file__),
                            "runs", "detect", "chess", "weights", "best.pt")
    best_dst = os.path.join(os.path.dirname(__file__), "models", "best.pt")

    if os.path.exists(best_src):
        import shutil
        os.makedirs(os.path.dirname(best_dst), exist_ok=True)
        shutil.copy2(best_src, best_dst)
        print(f"\nBest model copied to: {best_dst}")
        print("Ready to use — restart main.py and it will pick it up automatically.")
    else:
        print(f"\nWARNING: {best_src} not found after training.")


if __name__ == "__main__":
    main()
