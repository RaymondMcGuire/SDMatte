"""
Batch inference for a folder of images.

Outputs:
  output/images/<name>.png  (RGBA, matte applied, background transparent)
  output/mask/<name>.png    (grayscale mask 0-255)
"""

import argparse
import os
import sys
import warnings
from os.path import abspath, dirname, join, splitext

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate, get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo


def normalize_img(img):
    # map [0, 1] -> [-1, 1]
    return img * 2 - 1


def list_images(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = []
    for name in os.listdir(folder):
        if splitext(name)[1].lower() in exts:
            files.append(name)
    files.sort()
    return files


def build_person_detector(score_thresh):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    return DefaultPredictor(cfg)


def detect_person_bbox(predictor, img_bgr):
    outputs = predictor(img_bgr)
    instances = outputs.get("instances")
    if instances is None or len(instances) == 0:
        return None
    classes = instances.pred_classes.cpu().numpy()
    boxes = instances.pred_boxes.tensor.cpu().numpy()
    scores = instances.scores.cpu().numpy()
    person_idx = np.where(classes == 0)[0]
    if person_idx.size == 0:
        return None
    best = person_idx[np.argmax(scores[person_idx])]
    x1, y1, x2, y2 = boxes[best].tolist()
    return x1, y1, x2, y2


def run_single(model, img_bgr, bbox, resize):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = img_rgb.shape[:2]

    size = resize
    img_resized = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_LINEAR)

    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))

    sx = size / float(w)
    sy = size / float(h)
    rx1, rx2 = int(x1 * sx), int(x2 * sx)
    ry1, ry2 = int(y1 * sy), int(y2 * sy)
    rx1, rx2 = max(0, rx1), min(size, rx2)
    ry1, ry2 = max(0, ry1), min(size, ry2)

    bbox_mask = np.zeros((size, size), dtype=np.float32)
    if rx2 > rx1 and ry2 > ry1:
        bbox_mask[ry1:ry2, rx1:rx2] = 1.0

    bbox_coords = np.array([rx1 / size, ry1 / size, rx2 / size, ry2 / size], dtype=np.float32)

    image_t = F.to_tensor(img_resized).float()
    image_t = normalize_img(image_t)
    bbox_mask_t = F.to_tensor(bbox_mask).float()
    bbox_mask_t = normalize_img(bbox_mask_t)

    data = {
        "image": image_t.unsqueeze(0),
        "bbox_mask": bbox_mask_t.unsqueeze(0),
        "bbox_coords": torch.from_numpy(bbox_coords).unsqueeze(0),
        "is_trans": torch.tensor([0], dtype=torch.long),
    }

    with torch.no_grad():
        pred = model(data)
        output = pred.flatten(0, 2) * 255.0
        output = output.detach().cpu().numpy().astype(np.uint8)
        output = cv2.resize(output, (w, h), interpolation=cv2.INTER_LINEAR)
    return output


def main():
    # Ensure repo root is on sys.path so LazyConfig imports work under uv run.
    repo_root = abspath(join(dirname(__file__), os.pardir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resize", type=int, default=1024)
    parser.add_argument("--det-score", type=float, default=0.5)
    parser.add_argument("--on-miss", choices=["skip", "full"], default="full")
    parser.add_argument("--erode", type=int, default=0, help="Erode mask by N pixels to reduce halo")
    parser.add_argument("--premultiply", action="store_true", default=True, help="Premultiply RGB by alpha")
    parser.add_argument(
        "--unsafe-load",
        action="store_true",
        default=True,
        help="Allow torch.load(weights_only=False). Use only if you trust the checkpoint.",
    )
    args = parser.parse_args()

    warnings.filterwarnings(
        "ignore",
        message="`clean_up_tokenization_spaces` was not set.*",
        category=FutureWarning,
        module="transformers.tokenization_utils_base",
    )
    warnings.filterwarnings(
        "ignore",
        message="torch.meshgrid: in an upcoming release.*",
        category=UserWarning,
        module="torch.functional",
    )
    warnings.filterwarnings(
        "ignore",
        message="`scale` is deprecated and will be removed in version 1.0.0.*",
        category=FutureWarning,
        module="diffusers.models.unets.unet_2d_blocks",
    )

    if args.device != "cuda":
        raise ValueError("Only CUDA is supported by this repo's inference path.")

    torch.set_grad_enabled(False)
    cfg = LazyConfig.load(args.config_dir)
    model = instantiate(cfg.model)
    model.to(cfg.train.device)

    if args.unsafe_load:
        _orig_torch_load = torch.load

        def _torch_load_unsafe(*t_args, **t_kwargs):
            t_kwargs.setdefault("weights_only", False)
            return _orig_torch_load(*t_args, **t_kwargs)

        torch.load = _torch_load_unsafe

    DetectionCheckpointer(model).load(args.checkpoint_dir)
    model.eval()

    predictor = build_person_detector(args.det_score)

    input_dir = args.input_dir
    out_images = join(args.output_dir, "images")
    out_masks = join(args.output_dir, "mask")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_masks, exist_ok=True)

    files = list_images(input_dir)
    if not files:
        raise FileNotFoundError(f"No images found in {input_dir}")

    for name in files:
        img_path = join(input_dir, name)
        img_bgr = cv2.imread(img_path, 1)
        if img_bgr is None:
            continue

        det = detect_person_bbox(predictor, img_bgr)
        if det is None:
            if args.on_miss == "skip":
                continue
            h, w = img_bgr.shape[:2]
            det = (0, 0, w, h)

        mask = run_single(model, img_bgr, det, args.resize)
        if args.erode > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.erode, args.erode))
            mask = cv2.erode(mask, kernel, iterations=1)

        base = splitext(name)[0]
        mask_path = join(out_masks, base + ".png")
        cv2.imwrite(mask_path, mask)

        # Apply mask as alpha to original image (RGBA output)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        alpha = mask
        if args.premultiply:
            alpha_f = (alpha.astype(np.float32) / 255.0)[:, :, None]
            img_rgb = (img_rgb.astype(np.float32) * alpha_f).astype(np.uint8)
        rgba = np.dstack([img_rgb, alpha]).astype(np.uint8)
        out_path = join(out_images, base + ".png")
        cv2.imwrite(out_path, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))


if __name__ == "__main__":
    main()
