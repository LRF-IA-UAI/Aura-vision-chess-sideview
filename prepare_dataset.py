"""
Convert chess-pieces.ndjson to YOLO training format.

Downloads images from URLs and creates label .txt files.
Produces:
    datasets/chess-pieces/
        train/images/   train/labels/
        valid/images/   valid/labels/
        test/images/    test/labels/
        data.yaml
"""
import json
import os
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

NDJSON_PATH = os.path.join(os.path.dirname(__file__), "chess-pieces.ndjson")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "datasets", "chess-pieces")

SPLIT_MAP = {"train": "train", "val": "valid", "test": "test"}


def download_image(url, dest_path, retries=3):
    """Download a single image with retries."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            with open(dest_path, "wb") as f:
                f.write(data)
            return True
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            if attempt < retries - 1:
                time.sleep(1 * (attempt + 1))
            else:
                print(f"  FAILED: {os.path.basename(dest_path)}: {e}")
                return False


def write_label(boxes, label_path):
    """Write YOLO label file from annotation boxes."""
    with open(label_path, "w") as f:
        for box in boxes:
            cls_id = int(box[0])
            x_center, y_center, w, h = box[1], box[2], box[3], box[4]
            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")


def main():
    # Parse NDJSON
    print("Parsing NDJSON...")
    class_names = {}
    records = []

    with open(NDJSON_PATH, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj["type"] == "dataset":
                class_names = obj.get("class_names", {})
                continue
            if obj["type"] == "image":
                records.append(obj)

    print(f"  {len(records)} images found")
    print(f"  Classes: {class_names}")

    # Create directories
    for split_name in SPLIT_MAP.values():
        os.makedirs(os.path.join(OUTPUT_DIR, split_name, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split_name, "labels"), exist_ok=True)

    # Write data.yaml
    nc = len(class_names)
    names_list = [class_names[str(i)] for i in range(nc)]
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(OUTPUT_DIR)}\n")
        f.write("train: train/images\n")
        f.write("val: valid/images\n")
        f.write("test: test/images\n")
        f.write(f"nc: {nc}\n")
        f.write(f"names: {names_list}\n")
    print(f"  data.yaml written: {yaml_path}")

    # Prepare download tasks
    tasks = []
    skipped = 0
    no_split = 0
    for rec in records:
        split_raw = rec.get("split", "")
        split_dir = SPLIT_MAP.get(split_raw)
        if not split_dir:
            no_split += 1
            continue

        filename = rec["file"]
        stem = os.path.splitext(filename)[0]
        img_path = os.path.join(OUTPUT_DIR, split_dir, "images", filename)
        lbl_path = os.path.join(OUTPUT_DIR, split_dir, "labels", stem + ".txt")

        # Write label
        annotations = rec.get("annotations", {})
        boxes = annotations.get("boxes", [])
        write_label(boxes, lbl_path)

        # Queue download if image doesn't exist
        if os.path.exists(img_path):
            skipped += 1
        else:
            tasks.append((rec["url"], img_path))

    if no_split:
        print(f"  {no_split} records skipped (no split)")
    if skipped:
        print(f"  {skipped} images already downloaded, skipping")

    print(f"  {len(tasks)} images to download...")

    if not tasks:
        print("\nDone! All images already present.")
        print(f"Dataset ready at: {os.path.abspath(OUTPUT_DIR)}")
        return

    # Download with thread pool
    done = 0
    failed = 0
    total = len(tasks)
    start = time.time()

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(download_image, url, path): path for url, path in tasks}
        for future in as_completed(futures):
            result = future.result()
            if result:
                done += 1
            else:
                failed += 1
            count = done + failed
            if count % 100 == 0 or count == total:
                elapsed = time.time() - start
                rate = count / elapsed if elapsed > 0 else 0
                eta = (total - count) / rate if rate > 0 else 0
                print(f"  [{count}/{total}] ok={done} fail={failed} "
                      f"({rate:.1f} img/s, ETA {eta:.0f}s)")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Downloaded: {done}")
    print(f"  Failed: {failed}")
    print(f"  Dataset ready at: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
