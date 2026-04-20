import os
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from transformers import (
    DeformableDetrConfig,
    DeformableDetrForObjectDetection,
    DeformableDetrImageProcessor,
)

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    PYCOCOTOOLS_AVAILABLE = True
except Exception:
    PYCOCOTOOLS_AVAILABLE = False


# ============================================================
# Utility
# ============================================================

def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


# ============================================================
# Dataset
# ============================================================

class DigitCocoDataset(Dataset):
    """
    For your dataset:
    - train.json / valid.json are COCO-style
    - category_id starts from 1
    - internally we remap category ids to 0..K-1 for HF training
    - when saving pred.json, we map them back to original category_id
    """

    def __init__(
        self,
        img_dir,
        annotation_path=None,
        processor=None,
        cat_id_to_contig=None,
    ):
        self.img_dir = Path(img_dir)
        self.annotation_path = annotation_path
        self.processor = processor
        self.is_test = annotation_path is None

        if self.is_test:
            self.images = []
            for p in sorted(
                self.img_dir.glob("*.png"),
                key=lambda x: int(x.stem),
            ):
                self.images.append({
                    "id": int(p.stem),
                    "file_name": p.name,
                })
            self.annotations_by_image = {}
            self.cat_id_to_contig = cat_id_to_contig or {}
        else:
            data = load_json(annotation_path)
            self.images = sorted(data["images"], key=lambda x: x["id"])
            anns = data.get("annotations", [])
            categories = sorted(
                data.get("categories", []),
                key=lambda x: x["id"],
            )

            if cat_id_to_contig is None:
                original_cat_ids = [c["id"] for c in categories]
                self.cat_id_to_contig = {
                    cat_id: i
                    for i, cat_id in enumerate(original_cat_ids)
                }
            else:
                self.cat_id_to_contig = cat_id_to_contig

            self.annotations_by_image = defaultdict(list)
            for ann in anns:
                new_ann = dict(ann)
                new_ann["category_id"] = self.cat_id_to_contig[
                    ann["category_id"]
                ]
                self.annotations_by_image[ann["image_id"]].append(new_ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        image_id = img_info["id"]
        img_path = self.img_dir / img_info["file_name"]

        image = Image.open(img_path).convert("RGB")

        if self.is_test:
            return {
                "image_id": image_id,
                "image": image,
                "orig_size": image.size[::-1],  # (H, W)
            }

        anns = self.annotations_by_image[image_id]
        target = {
            "image_id": image_id,
            "annotations": [],
        }

        for ann in anns:
            x, y, w, h = ann["bbox"]
            area = ann.get("area", w * h)
            iscrowd = ann.get("iscrowd", 0)
            target["annotations"].append({
                "bbox": [x, y, w, h],
                "category_id": ann["category_id"],
                "area": area,
                "iscrowd": iscrowd,
            })

        encoding = self.processor(
            images=image,
            annotations=target,
            return_tensors="pt",
        )

        pixel_values = encoding["pixel_values"].squeeze(0)
        labels = encoding["labels"][0]

        return {
            "image_id": image_id,
            "pixel_values": pixel_values,
            "labels": labels,
        }


# ============================================================
# Processor compatibility helper
# ============================================================

def safe_pad(processor, pixel_values_list):
    try:
        return processor.pad(pixel_values_list, return_tensors="pt")
    except TypeError:
        pass

    try:
        return processor.pad(
            pixel_values_list,
            return_tensors="pt",
            input_data_format="channels_first",
        )
    except TypeError:
        pass

    max_h = max(x.shape[1] for x in pixel_values_list)
    max_w = max(x.shape[2] for x in pixel_values_list)
    batch_size = len(pixel_values_list)
    c = pixel_values_list[0].shape[0]

    pixel_values = torch.zeros(
        (batch_size, c, max_h, max_w),
        dtype=pixel_values_list[0].dtype,
    )
    pixel_mask = torch.zeros((batch_size, max_h, max_w), dtype=torch.long)

    for i, x in enumerate(pixel_values_list):
        _, h, w = x.shape
        pixel_values[i, :, :h, :w] = x
        pixel_mask[i, :h, :w] = 1

    return {"pixel_values": pixel_values, "pixel_mask": pixel_mask}


def train_collate_fn(batch, processor):
    pixel_values = [item["pixel_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    image_ids = [item["image_id"] for item in batch]

    padded = safe_pad(processor, pixel_values)
    return {
        "pixel_values": padded["pixel_values"],
        "pixel_mask": padded["pixel_mask"],
        "labels": labels,
        "image_ids": image_ids,
    }


def test_collate_fn(batch, processor):
    images = [item["image"] for item in batch]
    image_ids = [item["image_id"] for item in batch]
    orig_sizes = [item["orig_size"] for item in batch]

    enc = processor(images=images, return_tensors="pt")
    return {
        "pixel_values": enc["pixel_values"],
        "pixel_mask": enc.get("pixel_mask"),
        "image_ids": image_ids,
        "orig_sizes": orig_sizes,
    }


# ============================================================
# Model
# ============================================================

def build_model(num_classes, id2label, label2id, num_queries=30):
    """
    Requirement alignment:
    - Use Deformable DETR.
    - Use pretrained ResNet50 backbone.
    - Transformer detection part starts from scratch.

    Notes for local GPU:
    - We use a smaller config than the large default.
    - Two-stage is disabled to reduce memory.
    """

    config = DeformableDetrConfig(
        num_labels=num_classes,
        num_queries=num_queries,
        backbone="resnet50",
        use_pretrained_backbone=True,
        use_timm_backbone=True,
        num_feature_levels=4,
        encoder_layers=3,
        decoder_layers=3,
        encoder_attention_heads=8,
        decoder_attention_heads=8,
        encoder_ffn_dim=1024,
        decoder_ffn_dim=1024,
        d_model=256,
        auxiliary_loss=True,
        with_box_refine=True,
        two_stage=False,
        id2label=id2label,
        label2id=label2id,
    )

    model = DeformableDetrForObjectDetection(config)
    return model


def set_backbone_trainable(model, trainable: bool):
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = trainable


def build_optimizer(model, lr, lr_backbone, weight_decay):
    backbone_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "lr": lr},
            {"params": backbone_params, "lr": lr_backbone},
        ],
        weight_decay=weight_decay,
    )
    return optimizer


# ============================================================
# Training / Evaluation
# ============================================================

def move_labels_to_device(labels, device):
    out = []
    for target in labels:
        out.append({k: v.to(device) for k, v in target.items()})
    return out


def summarize_loss_dict(loss_dict):
    class_loss = 0.0
    bbox_loss = 0.0
    giou_loss = 0.0

    for k, v in loss_dict.items():
        kv = v.item()
        if "loss_ce" in k:
            class_loss += kv
        elif "loss_bbox" in k:
            bbox_loss += kv
        elif "loss_giou" in k:
            giou_loss += kv

    return class_loss, bbox_loss, giou_loss


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device,
    scaler,
    grad_clip=0.1,
    amp=True,
    accum_steps=1,
):
    model.train()
    total_loss = 0.0
    total_class_loss = 0.0
    total_bbox_loss = 0.0
    total_giou_loss = 0.0
    total_batches = 0

    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(data_loader, desc="Train", leave=False)

    for step, batch in enumerate(progress):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = (
            batch["pixel_mask"].to(device)
            if batch["pixel_mask"] is not None
            else None
        )
        labels = move_labels_to_device(batch["labels"], device)

        with torch.cuda.amp.autocast(enabled=amp):
            outputs = model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss_for_backward = loss / accum_steps

        scaler.scale(loss_for_backward).backward()

        if (step + 1) % accum_steps == 0:
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        class_loss, bbox_loss, giou_loss = summarize_loss_dict(
            outputs.loss_dict
        )

        total_loss += loss.item()
        total_class_loss += class_loss
        total_bbox_loss += bbox_loss
        total_giou_loss += giou_loss
        total_batches += 1

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            cls=f"{class_loss:.4f}",
            bbox=f"{bbox_loss:.4f}",
        )

    # flush gradients if last step not divisible by accum_steps
    if total_batches % accum_steps != 0:
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return (
        total_loss / max(total_batches, 1),
        total_class_loss / max(total_batches, 1),
        total_bbox_loss / max(total_batches, 1),
        total_giou_loss / max(total_batches, 1),
    )


@torch.no_grad()
def evaluate_loss(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_class_loss = 0.0
    total_bbox_loss = 0.0
    total_giou_loss = 0.0
    total_batches = 0

    for batch in tqdm(data_loader, desc="Valid", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = (
            batch["pixel_mask"].to(device)
            if batch["pixel_mask"] is not None
            else None
        )
        labels = move_labels_to_device(batch["labels"], device)

        outputs = model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels,
        )
        loss = outputs.loss
        class_loss, bbox_loss, giou_loss = summarize_loss_dict(
            outputs.loss_dict
        )

        total_loss += loss.item()
        total_class_loss += class_loss
        total_bbox_loss += bbox_loss
        total_giou_loss += giou_loss
        total_batches += 1

    return (
        total_loss / max(total_batches, 1),
        total_class_loss / max(total_batches, 1),
        total_bbox_loss / max(total_batches, 1),
        total_giou_loss / max(total_batches, 1),
    )


class ValidEvalWrapper(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        img_info = self.base.images[idx]
        image = Image.open(
            self.base.img_dir / img_info["file_name"]
        ).convert("RGB")
        orig_size = image.size[::-1]
        item["labels"]["orig_size"] = torch.tensor(orig_size, dtype=torch.long)
        return item


@torch.no_grad()
def coco_map_eval(
    model,
    data_loader,
    processor,
    device,
    contig_to_cat_id,
    ann_file,
    score_threshold=0.0,
):
    if not PYCOCOTOOLS_AVAILABLE:
        return None

    model.eval()
    preds = []

    for batch in tqdm(data_loader, desc="COCO eval", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        mask = batch["pixel_mask"]
        pixel_mask = mask.to(device) if mask is not None else None
        image_ids = batch["image_ids"]

        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        target_sizes = []
        for labels in batch["labels"]:
            size = labels["orig_size"].detach().cpu().tolist()
            target_sizes.append(size)
        target_sizes = torch.tensor(target_sizes, dtype=torch.long)

        results = processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=score_threshold,
        )

        for image_id, result in zip(image_ids, results):
            boxes = result["boxes"].cpu().tolist()
            scores = result["scores"].cpu().tolist()
            labels = result["labels"].cpu().tolist()

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                preds.append({
                    "image_id": int(image_id),
                    "category_id": int(contig_to_cat_id[int(label)]),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score),
                })

    coco_gt = COCO(ann_file)
    if len(preds) == 0:
        return 0.0

    coco_dt = coco_gt.loadRes(preds)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return float(coco_eval.stats[0])


# ============================================================
# Predict on test
# ============================================================

@torch.no_grad()
def predict_test(
    model,
    data_loader,
    processor,
    device,
    contig_to_cat_id,
    out_path,
    score_threshold=0.05,
):
    model.eval()
    predictions = []

    for batch in tqdm(data_loader, desc="Predict test"):
        pixel_values = batch["pixel_values"].to(device)
        mask = batch["pixel_mask"]
        pixel_mask = mask.to(device) if mask is not None else None
        image_ids = batch["image_ids"]
        orig_sizes = batch["orig_sizes"]

        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        target_sizes = torch.tensor(orig_sizes, dtype=torch.long)
        results = processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=score_threshold,
        )

        for image_id, result in zip(image_ids, results):
            boxes = result["boxes"].cpu().tolist()
            scores = result["scores"].cpu().tolist()
            labels = result["labels"].cpu().tolist()

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                predictions.append({
                    "image_id": int(image_id),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score),
                    "category_id": int(contig_to_cat_id[int(label)]),
                })

    save_json(predictions, out_path)
    print(f"Saved predictions to: {out_path}")


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="nycu-hw2-data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_queries", type=int, default=20)
    parser.add_argument("--score_threshold", type=float, default=0.001)
    parser.add_argument("--map_threshold", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="outputs_deformable")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--eval_map", action="store_true")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=5)
    parser.add_argument("--accum_steps", type=int, default=2)
    parser.add_argument("--predict_only", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    root = Path(args.root)
    train_dir = root / "train"
    valid_dir = root / "valid"
    test_dir = root / "test"
    train_json = root / "train.json"
    valid_json = root / "valid.json"

    if not all([train_dir.exists(), valid_dir.exists(), test_dir.exists()]):
        raise FileNotFoundError(
            "Please check your folder structure under nycu-hw2-data/"
        )
    if not train_json.exists() or not valid_json.exists():
        raise FileNotFoundError("Please check train.json and valid.json")

    # Local GPU friendly image size.
    processor = DeformableDetrImageProcessor(
        format="coco_detection",
        do_resize=True,
        size={"shortest_edge": 384, "longest_edge": 384},
        do_rescale=True,
        do_normalize=True,
    )

    train_meta = load_json(str(train_json))
    categories = sorted(train_meta["categories"], key=lambda x: x["id"])
    original_cat_ids = [c["id"] for c in categories]

    cat_id_to_contig = {cat_id: i for i, cat_id in enumerate(original_cat_ids)}
    contig_to_cat_id = {i: cat_id for cat_id, i in cat_id_to_contig.items()}

    num_classes = len(original_cat_ids)
    id2label = {i: str(contig_to_cat_id[i]) for i in range(num_classes)}
    label2id = {v: k for k, v in id2label.items()}

    train_dataset = DigitCocoDataset(
        img_dir=str(train_dir),
        annotation_path=str(train_json),
        processor=processor,
        cat_id_to_contig=cat_id_to_contig,
    )

    valid_dataset = DigitCocoDataset(
        img_dir=str(valid_dir),
        annotation_path=str(valid_json),
        processor=processor,
        cat_id_to_contig=cat_id_to_contig,
    )

    valid_eval_dataset = ValidEvalWrapper(valid_dataset)

    test_dataset = DigitCocoDataset(
        img_dir=str(test_dir),
        annotation_path=None,
        processor=processor,
        cat_id_to_contig=cat_id_to_contig,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda b: train_collate_fn(b, processor),
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda b: train_collate_fn(b, processor),
    )

    valid_eval_loader = DataLoader(
        valid_eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda b: train_collate_fn(b, processor),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda b: test_collate_fn(b, processor),
    )

    model = build_model(
        num_classes=num_classes,
        id2label=id2label,
        label2id=label2id,
        num_queries=args.num_queries,
    ).to(device)

    best_mAP = 0.0
    map_score = -1.0
    start_epoch = 1

    if args.predict_only:
        print("Predict-only mode: skip training and generate pred.json")

        ckpt = torch.load(args.predict_only, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        best_map = ckpt.get("best_mAP", "N/A")

        print(
            f"Loaded model from {args.predict_only} "
            f"with the mAP score of {best_map}"
        )

        pred_path = os.path.join(args.save_dir, "pred.json")
        predict_test(
            model=model,
            data_loader=test_loader,
            processor=processor,
            device=device,
            contig_to_cat_id=contig_to_cat_id,
            out_path=pred_path,
            score_threshold=args.score_threshold,
        )

        print("Done.")
        return

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if "best_mAP" in ckpt:
            best_mAP = ckpt["best_mAP"]

        start_epoch = ckpt["epoch"] + 1
        best_map_str = f"{best_mAP:.4f}"

        print(
            f"Resumed from {args.resume}, "
            f"start epoch = {start_epoch}, "
            f"best mAP = {best_map_str}"
        )
        set_backbone_trainable(model, True)
        optimizer = build_optimizer(
            model,
            args.lr,
            args.lr_backbone,
            args.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=15,
            gamma=0.1,
        )

    if args.freeze_backbone_epochs > 0:
        set_backbone_trainable(model, False)
        print(
            f"Backbone frozen for first "
            f"{args.freeze_backbone_epochs} epochs."
        )

    optimizer = build_optimizer(
        model,
        args.lr,
        args.lr_backbone,
        args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=15,
        gamma=0.1,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_ckpt_path = os.path.join(args.save_dir, "best_dd_1e4.pth")
    best_ckpt_path_unfreeze = os.path.join(
        args.save_dir,
        "best_dd_1e4_unfreeze.pth",
    )

    backbone_unfrozen = (args.freeze_backbone_epochs == 0)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n========== Epoch {epoch}/{args.epochs} ==========")
        begin = time.time()

        if epoch == args.freeze_backbone_epochs + 1 and not backbone_unfrozen:
            set_backbone_trainable(model, True)
            optimizer = build_optimizer(
                model,
                args.lr,
                args.lr_backbone,
                args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=15,
                gamma=0.1,
            )
            backbone_unfrozen = True
            print("Backbone unfrozen. Now training full model.")

        train_loss, train_cls, train_bbox, train_giou = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            grad_clip=0.1,
            amp=(device.type == "cuda"),
            accum_steps=args.accum_steps,
        )

        valid_loss, valid_cls, valid_bbox, valid_giou = evaluate_loss(
            model=model,
            data_loader=valid_loader,
            device=device,
        )

        scheduler.step()

        print(
            f"Train loss: {train_loss:.4f} | "
            f"class loss: {train_cls:.4f} | "
            f"bbox loss: {train_bbox:.4f} | "
            f"giou loss: {train_giou:.4f}"
        )

        print(
            f"Valid loss: {valid_loss:.4f} | "
            f"class loss: {valid_cls:.4f} | "
            f"bbox loss: {valid_bbox:.4f} | "
            f"giou loss: {valid_giou:.4f}"
        )

        if args.eval_map:
            if PYCOCOTOOLS_AVAILABLE:
                map_score = coco_map_eval(
                    model=model,
                    data_loader=valid_eval_loader,
                    processor=processor,
                    device=device,
                    contig_to_cat_id=contig_to_cat_id,
                    ann_file=str(valid_json),
                    score_threshold=args.map_threshold,
                )
                if map_score is not None:
                    print(f"mAP@[0.5:0.95]: {map_score:.4f}")
            else:
                print("pycocotools is not installed, skip mAP evaluation.")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_mAP": map_score,
            "cat_id_to_contig": cat_id_to_contig,
            "contig_to_cat_id": contig_to_cat_id,
        }
        torch.save(ckpt, os.path.join(args.save_dir, "last.pth"))

        if map_score > best_mAP:
            best_mAP = map_score
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_mAP": best_mAP,
                "cat_id_to_contig": cat_id_to_contig,
                "contig_to_cat_id": contig_to_cat_id,
            }
            if backbone_unfrozen:
                torch.save(ckpt, best_ckpt_path_unfreeze)
                print(f"Best model saved to {best_ckpt_path_unfreeze}")
            else:
                torch.save(ckpt, best_ckpt_path)
                print(f"Best model saved to {best_ckpt_path}")

        print(f"Epoch time: {time.time() - begin:.1f}s")

    print("\nTraining finished.")
    print(f"Loading best checkpoint from {best_ckpt_path}")
    best_ckpt = torch.load(best_ckpt_path, map_location="cpu")
    model.load_state_dict(best_ckpt["model"])
    model.to(device)

    pred_path = os.path.join(args.save_dir, "pred.json")
    predict_test(
        model=model,
        data_loader=test_loader,
        processor=processor,
        device=device,
        contig_to_cat_id=contig_to_cat_id,
        out_path=pred_path,
        score_threshold=args.score_threshold,
    )

    print("Done.")


if __name__ == "__main__":
    main()
