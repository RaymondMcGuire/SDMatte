"""
Single-image inference for SDMatte / LiteSDMatte.

Example:
python script/infer_single.py ^
  --config-dir configs/SDMatte.py ^
  --checkpoint-dir SDMatte/SDMatte.pth ^
  --image path/to/image.jpg ^
  --bbox 100,50,800,900 ^
  --output output_mask.png
"""

import argparse
import os
import sys
import warnings
from os.path import dirname, abspath, join

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


def normalize_img(img):
    # map [0, 1] -> [-1, 1]
    return img * 2 - 1


def parse_bbox(bbox_str):
    parts = bbox_str.split(",")
    if len(parts) != 4:
        raise ValueError("--bbox must be 'x1,y1,x2,y2'")
    x1, y1, x2, y2 = [float(p) for p in parts]
    return x1, y1, x2, y2


def detect_person_bbox(img_bgr, score_thresh=0.5):
    # COCO class 0 is person
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

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

    # pick highest score person (or largest area if scores tie)
    best = person_idx[np.argmax(scores[person_idx])]
    x1, y1, x2, y2 = boxes[best].tolist()
    return x1, y1, x2, y2


def main():
    # Ensure repo root is on sys.path so LazyConfig imports work under uv run.
    repo_root = abspath(join(dirname(__file__), os.pardir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--bbox", required=False, help="x1,y1,x2,y2 in pixel coords (original image)")
    parser.add_argument("--auto-bbox", action="store_true", help="auto-detect person bbox using detectron2")
    parser.add_argument("--det-score", type=float, default=0.5, help="person detector score threshold")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resize", type=int, default=1024, help="inference size (square)")
    parser.add_argument(
        "--unsafe-load",
        action="store_true",
        default=True,
        help="Allow torch.load(weights_only=False). Use only if you trust the checkpoint.",
    )
    args = parser.parse_args()

    # Reduce noisy warnings from upstream deps.
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

    # load model
    torch.set_grad_enabled(False)
    cfg = LazyConfig.load(args.config_dir)
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    # Allow ListConfig in torch.load (PyTorch >=2.6 uses weights_only=True by default).
    if args.unsafe_load:
        # Force legacy (unsafe) loading for older checkpoints with custom objects.
        _orig_torch_load = torch.load

        def _torch_load_unsafe(*t_args, **t_kwargs):
            t_kwargs.setdefault("weights_only", False)
            return _orig_torch_load(*t_args, **t_kwargs)

        torch.load = _torch_load_unsafe
    DetectionCheckpointer(model).load(args.checkpoint_dir)
    model.eval()

    if not args.bbox and not args.auto_bbox:
        raise ValueError("Either --bbox or --auto-bbox is required.")

    # load image
    img_bgr = cv2.imread(args.image, 1)
    if img_bgr is None:
        raise FileNotFoundError(args.image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = img_rgb.shape[:2]

    # resize image
    size = args.resize
    img_resized = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_LINEAR)

    # bbox mask (resized space)
    if args.auto_bbox:
        det = detect_person_bbox(img_bgr, score_thresh=args.det_score)
        if det is None:
            raise RuntimeError("No person detected. Try lowering --det-score or provide --bbox.")
        x1, y1, x2, y2 = det
    else:
        x1, y1, x2, y2 = parse_bbox(args.bbox)
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))

    # scale bbox to resized space
    sx = size / float(w)
    sy = size / float(h)
    rx1, rx2 = int(x1 * sx), int(x2 * sx)
    ry1, ry2 = int(y1 * sy), int(y2 * sy)
    rx1, rx2 = max(0, rx1), min(size, rx2)
    ry1, ry2 = max(0, ry1), min(size, ry2)

    bbox_mask = np.zeros((size, size), dtype=np.float32)
    if rx2 > rx1 and ry2 > ry1:
        bbox_mask[ry1:ry2, rx1:rx2] = 1.0

    # coords normalized to resized image
    bbox_coords = np.array(
        [rx1 / size, ry1 / size, rx2 / size, ry2 / size],
        dtype=np.float32,
    )

    # build data dict (matching dataset transform output)
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

    # forward
    with torch.no_grad():
        pred = model(data)
        alpha = pred.flatten(0, 2) * 255.0
        alpha = alpha.detach().cpu().numpy().astype(np.uint8)
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)

    # Save RGBA (foreground with alpha matte)
    os.makedirs(dirname(args.output) or ".", exist_ok=True)
    img_rgb_u8 = (img_rgb * 255.0).astype(np.uint8)
    rgba = np.dstack([img_rgb_u8, alpha]).astype(np.uint8)
    cv2.imwrite(args.output, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))


if __name__ == "__main__":
    main()
