import argparse
import copy
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader
from tqdm import tqdm

from yolov8nano.data.yolo_dataset import YOLODetectionDataset, detection_collate_fn
from yolov8nano.losses.yolov8_loss import YOLOv8Loss
from yolov8nano.models.yolov8 import YOLOv8Nano
from yolov8nano.utils.ema import ModelEMA
from yolov8nano.utils.inference import decode_predictions, postprocess_to_original
from yolov8nano.utils.metrics import DetectionMetric, labels_to_original
from yolov8nano.utils.plots import plot_detection_curves, plot_per_class_results, plot_results


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8-nano style detector from scratch.")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_dataloader(config, split: str):
    dataset_cfg = config["dataset"]
    train_mode = split == "train"
    dataset = YOLODetectionDataset(
        image_dir=dataset_cfg[f"{split}_images"],
        label_dir=dataset_cfg[f"{split}_labels"],
        image_size=config["train"]["image_size"],
        augment=train_mode,
        augment_cfg=config.get("augment", {}) if train_mode else None,
        class_names=dataset_cfg.get("names", []),
    )
    return DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=train_mode,
        num_workers=config["train"]["num_workers"],
        pin_memory=True,
        persistent_workers=config["train"]["num_workers"] > 0 and config["train"].get("persistent_workers", True),
        prefetch_factor=config["train"].get("prefetch_factor", 2) if config["train"]["num_workers"] > 0 else None,
        collate_fn=detection_collate_fn,
        drop_last=train_mode,
    )


def build_optimizer(model, config):
    optimizer_name = str(config["train"].get("optimizer", "adamw")).lower()
    momentum = config["train"].get("momentum", 0.937)
    nbs = config["train"].get("nbs", 64)
    batch_size = config["train"]["batch_size"]
    base_accumulate = max(round(nbs / batch_size), 1)
    scaled_weight_decay = config["train"]["weight_decay"] * batch_size * base_accumulate / nbs

    norm_layers = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        torch.nn.GroupNorm,
        torch.nn.LayerNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
    )
    decay_params = []
    norm_params = []
    bias_params = []

    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if name == "bias":
                bias_params.append(param)
            elif isinstance(module, norm_layers):
                norm_params.append(param)
            else:
                decay_params.append(param)

    lr = config["train"]["lr"]
    if optimizer_name == "adamw":
        optimizer = AdamW(bias_params, lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif optimizer_name == "sgd":
        optimizer = SGD(bias_params, lr=lr, momentum=momentum, nesterov=True)
    else:
        raise ValueError(f"unsupported optimizer: {optimizer_name}")
    optimizer.add_param_group({"params": decay_params, "weight_decay": scaled_weight_decay})
    optimizer.add_param_group({"params": norm_params, "weight_decay": 0.0})
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])
    return optimizer, base_accumulate, scaled_weight_decay, optimizer_name


def build_scheduler(optimizer, config):
    epochs = max(int(config["train"]["epochs"]), 1)
    lrf = float(config["train"].get("lrf", 0.01))
    cos_lr = bool(config["train"].get("cos_lr", False))

    if cos_lr:
        def lr_lambda(epoch):
            if epochs <= 1:
                return lrf
            progress = epoch / max(epochs - 1, 1)
            return ((1.0 - np.cos(progress * np.pi)) / 2.0) * (lrf - 1.0) + 1.0
    else:
        def lr_lambda(epoch):
            if epochs <= 1:
                return lrf
            return (1.0 - epoch / max(epochs - 1, 1)) * (1.0 - lrf) + lrf

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def count_parameters(model) -> int:
    return sum(param.numel() for param in model.parameters())


def count_gradients(model) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def estimate_gflops(model, image_size: int, device: torch.device):
    dummy = torch.zeros((1, 3, image_size, image_size), device=device)
    was_training = model.training
    model.eval()
    try:
        from thop import profile as thop_profile

        macs, _ = thop_profile(model, inputs=(dummy,), verbose=False)
        if macs and macs > 0:
            model.train(was_training)
            return (macs * 2.0) / 1e9
    except Exception:
        pass

    try:
        activities = [ProfilerActivity.CPU]
        if device.type == "cuda" and torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        with profile(activities=activities, with_flops=True) as prof:
            with torch.no_grad():
                model(dummy)
        total_flops = sum(getattr(event, "flops", 0) for event in prof.key_averages())
        if total_flops > 0:
            model.train(was_training)
            return total_flops / 1e9
    except Exception:
        pass

    model.train(was_training)
    return None


def get_gpu_mem(device: torch.device) -> str:
    if device.type != "cuda":
        return "0.00G"
    mem_gb = torch.cuda.memory_reserved(device) / (1024**3)
    return f"{mem_gb:.2f}G"


def format_seconds(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def should_close_aug(epoch: int, epochs: int, close_aug_epochs: int) -> bool:
    if close_aug_epochs <= 0:
        return False
    return epoch >= max(epochs - close_aug_epochs, 0)


def should_validate_epoch(epoch: int, epochs: int, val_interval: int, final_val_epochs: int) -> bool:
    if epoch >= max(epochs - final_val_epochs, 0):
        return True
    return (epoch + 1) % max(val_interval, 1) == 0


def skipped_val_stats(num_classes: int):
    nan_array = np.full(num_classes, np.nan, dtype=np.float32)
    return {
        "loss": float("nan"),
        "box_loss": float("nan"),
        "cls_loss": float("nan"),
        "dfl_loss": float("nan"),
        "mp": float("nan"),
        "mr": float("nan"),
        "map50": float("nan"),
        "map": float("nan"),
        "curves": None,
        "per_class": {
            "precision": nan_array.copy(),
            "recall": nan_array.copy(),
            "ap50": nan_array.copy(),
            "ap5095": nan_array.copy(),
            "valid": np.zeros(num_classes, dtype=bool),
        },
        "elapsed": 0.0,
        "validated": False,
    }


def get_current_lr(optimizer) -> float:
    if len(optimizer.param_groups) >= 2:
        return optimizer.param_groups[1]["lr"]
    return optimizer.param_groups[0]["lr"]


def print_epoch_table_header():
    print("-" * 154)
    print(
        f"{'Epoch':>8} {'GPU_mem':>8} {'box_loss':>10} {'cls_loss':>10} {'dfl_loss':>10} "
        f"{'Instances':>10} {'Size':>8} {'val_loss':>10} {'P':>8} {'R':>8} {'mAP50':>8} {'mAP50-95':>10} "
        f"{'lr':>12} {'train_t':>8} {'val_t':>8} {'time':>8}"
    )
    print("-" * 154)


def print_train_overview(config, model, train_loader, val_loader, device):
    print("Ultralytics-style Training Log")
    print(f"Model: YOLOv8-nano style detector")
    gflops = estimate_gflops(model, int(config["train"]["image_size"]), device)
    gflops_text = f" | GFLOPs@{config['train']['image_size']}: {gflops:.2f}" if gflops is not None else ""
    print(
        f"Classes: {config['model']['num_classes']} | "
        f"Params: {count_parameters(model):,} | "
        f"Gradients: {count_gradients(model):,}"
        f"{gflops_text}"
    )
    print(
        f"Train images: {len(train_loader.dataset)} | "
        f"Val images: {len(val_loader.dataset)} | "
        f"Batch size: {config['train']['batch_size']} | "
        f"Image size: {config['train']['image_size']}"
    )
    nbs = config["train"].get("nbs", 64)
    base_accumulate = max(round(nbs / config["train"]["batch_size"]), 1)
    print(
        f"Device: {device} | AMP: {config['train'].get('amp', True)} | "
        f"Optimizer: {str(config['train'].get('optimizer', 'adamw')).upper()} | "
        f"EMA: {config['train'].get('ema', True)} | "
        f"Multi-scale: {config['train'].get('multi_scale', False)} | "
        f"cuDNN benchmark: {config['train'].get('cudnn_benchmark', True)} | "
        f"Warmup: {config['train'].get('warmup_epochs', 0)} ep | "
        f"Accumulate: {base_accumulate} | "
        f"Persistent workers: {config['train'].get('persistent_workers', True)} | "
        f"Prefetch: {config['train'].get('prefetch_factor', 2)} | "
        f"close_aug: {config['train'].get('close_aug', 0)} | "
        f"val_interval: {config['train'].get('val_interval', 1)} | "
        f"final_val_epochs: {config['train'].get('final_val_epochs', 10)}"
    )


def ensure_results_file(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "results.csv"
    if not results_file.exists():
        with results_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "box_loss",
                    "cls_loss",
                    "dfl_loss",
                    "precision",
                    "recall",
                    "map50",
                    "map50_95",
                    "val_loss",
                    "lr",
                    "imgsz",
                ]
            )
    return results_file


def per_class_results_path(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "results_per_class.csv"


def append_results_row(results_file: Path, epoch: int, train_stats, val_stats, lr: float, imgsz: int):
    with results_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                epoch + 1,
                f"{train_stats['box_loss']:.6f}",
                f"{train_stats['cls_loss']:.6f}",
                f"{train_stats['dfl_loss']:.6f}",
                f"{val_stats['mp']:.6f}",
                f"{val_stats['mr']:.6f}",
                f"{val_stats['map50']:.6f}",
                f"{val_stats['map']:.6f}",
                f"{val_stats['loss']:.6f}",
                f"{lr:.8f}",
                imgsz,
            ]
        )


def write_per_class_results(results_file: Path, epoch: int, val_stats, class_names):
    per_class = val_stats["per_class"]
    names = list(class_names or [])
    with results_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "class_id", "class_name", "precision", "recall", "map50", "map50_95", "valid"])
        writer.writerow(
            [
                epoch + 1,
                "all",
                "all",
                f"{val_stats['mp']:.6f}",
                f"{val_stats['mr']:.6f}",
                f"{val_stats['map50']:.6f}",
                f"{val_stats['map']:.6f}",
                int(bool(np.any(per_class["valid"]))),
            ]
        )
        for class_idx in range(len(per_class["precision"])):
            class_name = names[class_idx] if class_idx < len(names) else str(class_idx)
            writer.writerow(
                [
                    epoch + 1,
                    class_idx,
                    class_name,
                    f"{per_class['precision'][class_idx]:.6f}",
                    f"{per_class['recall'][class_idx]:.6f}",
                    f"{per_class['ap50'][class_idx]:.6f}",
                    f"{per_class['ap5095'][class_idx]:.6f}",
                    int(bool(per_class["valid"][class_idx])),
                ]
            )


def print_per_class_metrics(epoch: int, epochs: int, val_stats, class_names):
    per_class = val_stats["per_class"]
    names = list(class_names or [])
    print(f"Per-class metrics for best.pt ({epoch + 1}/{epochs}):")
    print(f"{'class':>16} {'P':>8} {'R':>8} {'mAP50':>8} {'mAP50-95':>10} {'valid':>8}")
    print(
        f"{'all':>16.16} "
        f"{val_stats['mp']:>8.4f} "
        f"{val_stats['mr']:>8.4f} "
        f"{val_stats['map50']:>8.4f} "
        f"{val_stats['map']:>10.4f} "
        f"{str(bool(np.any(per_class['valid']))):>8}"
    )
    for class_idx in range(len(per_class["precision"])):
        class_name = names[class_idx] if class_idx < len(names) else str(class_idx)
        print(
            f"{class_name:>16.16} "
            f"{per_class['precision'][class_idx]:>8.4f} "
            f"{per_class['recall'][class_idx]:>8.4f} "
            f"{per_class['ap50'][class_idx]:>8.4f} "
            f"{per_class['ap5095'][class_idx]:>10.4f} "
            f"{str(bool(per_class['valid'][class_idx])):>8}"
        )


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
        },
        path,
    )


def multi_scale_resize(images, targets, image_size: int, scale_range, stride: int):
    min_scale, max_scale = scale_range
    target_size = int(round(random.uniform(min_scale, max_scale) * image_size / stride) * stride)
    if target_size == image_size:
        return images, targets
    scale_factor = target_size / image_size
    images = F.interpolate(images, size=(target_size, target_size), mode="bilinear", align_corners=False)
    scaled_targets = []
    for target in targets:
        target = target.clone()
        if target.numel():
            target[:, 1:5] *= scale_factor
        scaled_targets.append(target)
    return images, scaled_targets


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    amp_enabled,
    config,
    epoch,
    num_batches,
    warmup_iters,
    base_accumulate,
    target_lrs,
    enable_multi_scale=True,
    ema=None,
):
    model.train()
    running = {"loss": 0.0, "box_loss": 0.0, "cls_loss": 0.0, "dfl_loss": 0.0}
    instance_count = 0
    current_imgsz = config["train"]["image_size"]
    current_accumulate = base_accumulate
    last_opt_step = epoch * num_batches - 1
    momentum = config["train"].get("momentum", 0.937)
    warmup_momentum = config["train"].get("warmup_momentum", 0.8)
    warmup_bias_lr = config["train"].get("warmup_bias_lr", 0.1)
    pbar = tqdm(loader, desc="train", leave=False, dynamic_ncols=True)
    optimizer.zero_grad(set_to_none=True)
    for batch_idx, batch in enumerate(pbar):
        ni = batch_idx + num_batches * epoch
        images = batch["images"].to(device, non_blocking=True)
        targets = [t.to(device) for t in batch["targets"]]
        batch_instances = sum(int(t.shape[0]) for t in targets)
        instance_count += batch_instances
        if enable_multi_scale and config["train"].get("multi_scale", False):
            images, targets = multi_scale_resize(
                images,
                targets,
                image_size=config["train"]["image_size"],
                scale_range=tuple(config["train"].get("multi_scale_range", [0.5, 1.5])),
                stride=config["train"].get("multi_scale_stride", 32),
            )
        current_imgsz = int(images.shape[-1])

        if warmup_iters > 0 and ni <= warmup_iters:
            current_accumulate = max(1, int(round(np.interp(ni, [0, warmup_iters], [1, base_accumulate]))))
            for group_idx, group in enumerate(optimizer.param_groups):
                warmup_lr = warmup_bias_lr if group_idx == 0 else 0.0
                group["lr"] = float(np.interp(ni, [0, warmup_iters], [warmup_lr, target_lrs[group_idx]]))
                if "momentum" in group:
                    group["momentum"] = float(np.interp(ni, [0, warmup_iters], [warmup_momentum, momentum]))
                elif "betas" in group:
                    _, beta2 = group["betas"]
                    group["betas"] = (
                        float(np.interp(ni, [0, warmup_iters], [warmup_momentum, momentum])),
                        beta2,
                    )
        else:
            current_accumulate = base_accumulate

        with autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            loss_items = criterion(outputs, targets)
            loss = loss_items["loss"]
        scaler.scale(loss).backward()

        if (ni - last_opt_step) >= current_accumulate or (batch_idx + 1) == num_batches:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            last_opt_step = ni
            if ema is not None:
                ema.update(model)

        for key in running:
            running[key] += loss_items[key].item()
        pbar.set_postfix(
            gpu_mem=get_gpu_mem(device),
            box=f"{loss_items['box_loss'].item():.4f}",
            cls=f"{loss_items['cls_loss'].item():.4f}",
            dfl=f"{loss_items['dfl_loss'].item():.4f}",
            instances=batch_instances,
            imgsz=current_imgsz,
            lr=f"{get_current_lr(optimizer):.6f}",
            acc=current_accumulate,
        )

    num_batches = max(len(loader), 1)
    results = {k: v / num_batches for k, v in running.items()}
    results["instances"] = instance_count
    results["imgsz"] = current_imgsz
    return results


@torch.no_grad()
def validate(model, loader, criterion, device, config):
    val_start = time.time()
    model.eval()
    fast_val = bool(config.get("val", {}).get("fast_val", False))
    running = {"loss": 0.0, "box_loss": 0.0, "cls_loss": 0.0, "dfl_loss": 0.0}
    metric = DetectionMetric(config["model"]["num_classes"])
    pbar = tqdm(loader, desc="val", leave=False, dynamic_ncols=True)
    amp_enabled = config["train"].get("amp", True) and device.type == "cuda"
    for batch in pbar:
        images = batch["images"].to(device, non_blocking=True)
        targets = [t.to(device) for t in batch["targets"]]
        with autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            detections = decode_predictions(
                outputs,
                reg_max=config["model"].get("reg_max", 16),
                strides=config["model"].get("strides", [8, 16, 32]),
                conf_threshold=config["val"].get("conf_threshold", 0.001),
                iou_threshold=config["val"].get("nms_iou_threshold", 0.65),
                max_det=config["val"].get("max_det", 300),
            )
            if not fast_val:
                loss_items = criterion(outputs, targets)
        if not fast_val:
            for key in running:
                running[key] += loss_items[key].item()
        for det, label, gain, pad, shape in zip(
            detections, batch["targets"], batch["gains"], batch["pads"], batch["shapes"]
        ):
            det = postprocess_to_original(det.cpu(), gain, pad, shape)
            label = labels_to_original(label.float(), gain, pad, shape)
            metric.update(det, label)
        if fast_val:
            pbar.set_postfix(metrics="fast")
        else:
            pbar.set_postfix(
                val_loss=f"{loss_items['loss'].item():.4f}",
                box=f"{loss_items['box_loss'].item():.4f}",
                cls=f"{loss_items['cls_loss'].item():.4f}",
                dfl=f"{loss_items['dfl_loss'].item():.4f}",
            )

    num_batches = max(len(loader), 1)
    if fast_val:
        results = {key: float("nan") for key in running}
    else:
        results = {k: v / num_batches for k, v in running.items()}
    results.update(metric.compute())
    results["elapsed"] = time.time() - val_start
    results["validated"] = True
    return results


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["train"].get("seed", 42))

    requested_device = config["train"].get("device", "cuda")
    device = torch.device(requested_device if requested_device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(config["train"].get("cudnn_benchmark", True))

    model = YOLOv8Nano(
        num_classes=config["model"]["num_classes"],
        width_mult=config["model"].get("width_mult", 0.25),
        depth_mult=config["model"].get("depth_mult", 0.33),
        reg_max=config["model"].get("reg_max", 16),
    ).to(device)
    criterion = YOLOv8Loss(
        num_classes=config["model"]["num_classes"],
        reg_max=config["model"].get("reg_max", 16),
        strides=tuple(config["model"].get("strides", [8, 16, 32])),
        box_weight=config.get("loss", {}).get("box_weight", 7.5),
        cls_weight=config.get("loss", {}).get("cls_weight", 1.0),
        dfl_weight=config.get("loss", {}).get("dfl_weight", 1.5),
        assigner_topk=config.get("assigner", {}).get("topk", 10),
        assigner_alpha=config.get("assigner", {}).get("alpha", 0.5),
        assigner_beta=config.get("assigner", {}).get("beta", 6.0),
    )
    optimizer, base_accumulate, scaled_weight_decay, optimizer_name = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    scaler = GradScaler(enabled=config["train"].get("amp", True) and device.type == "cuda")
    ema = ModelEMA(model, decay=config["train"].get("ema_decay", 0.9999)) if config["train"].get("ema", True) else None

    train_loader = build_dataloader(config, "train")
    val_loader = build_dataloader(config, "val")

    out_dir = Path(config["train"]["out_dir"])
    results_file = ensure_results_file(out_dir)
    per_class_results_file = per_class_results_path(out_dir)
    print_train_overview(config, model, train_loader, val_loader, device)
    best_map = -1.0
    best_epoch = None
    best_val_stats = None
    best_curves = None
    epochs = int(config["train"]["epochs"])
    close_aug_epochs = int(config["train"].get("close_aug", 0))
    val_interval = int(config["train"].get("val_interval", 1))
    final_val_epochs = int(config["train"].get("final_val_epochs", 10))
    num_batches = max(len(train_loader), 1)
    warmup_iters = max(round(config["train"].get("warmup_epochs", 0.0) * num_batches), 0)
    print(
        f"Optimizer ready ({optimizer_name}) | scaled_weight_decay={scaled_weight_decay:.6f} | "
        f"base_accumulate={base_accumulate} | warmup_iters={warmup_iters}"
    )
    for epoch in range(epochs):
        epoch_start = time.time()
        close_aug_active = should_close_aug(epoch, epochs, close_aug_epochs)
        if hasattr(train_loader.dataset, "set_close_aug"):
            train_loader.dataset.set_close_aug(close_aug_active)
        target_lrs = scheduler.get_last_lr()
        train_stats = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            scaler.is_enabled(),
            config,
            epoch,
            num_batches,
            warmup_iters,
            base_accumulate,
            target_lrs,
            enable_multi_scale=not close_aug_active,
            ema=ema,
        )
        train_elapsed = time.time() - epoch_start
        if ema is not None:
            ema.update_attr(model)
        eval_model = ema.ema if ema is not None else model
        validate_this_epoch = should_validate_epoch(epoch, epochs, val_interval, final_val_epochs)
        if validate_this_epoch:
            val_stats = validate(eval_model, val_loader, criterion, device, config)
        else:
            val_stats = skipped_val_stats(config["model"]["num_classes"])
        current_lr = get_current_lr(optimizer)
        scheduler.step()
        val_elapsed = val_stats.get("elapsed", 0.0)
        elapsed = format_seconds(train_elapsed + val_elapsed)
        print_epoch_table_header()
        print(
            f"{epoch + 1:>4d}/{epochs:<3d} "
            f"{get_gpu_mem(device):>8} "
            f"{train_stats['box_loss']:>10.4f} "
            f"{train_stats['cls_loss']:>10.4f} "
            f"{train_stats['dfl_loss']:>10.4f} "
            f"{train_stats['instances']:>10d} "
            f"{train_stats['imgsz']:>8d} "
            f"{val_stats['loss']:>10.4f} "
            f"{val_stats['mp']:>8.4f} "
            f"{val_stats['mr']:>8.4f} "
            f"{val_stats['map50']:>8.4f} "
            f"{val_stats['map']:>10.4f} "
            f"{current_lr:>12.8f} "
            f"{format_seconds(train_elapsed):>8} "
            f"{format_seconds(val_elapsed):>8} "
            f"{elapsed:>8}"
        )
        if close_aug_active:
            print(f"close_aug active: disabled random perspective / HSV / multi_scale at epoch {epoch + 1}/{epochs}")
        if not validate_this_epoch:
            print(
                f"validation skipped at epoch {epoch + 1}/{epochs} "
                f"(val_interval={val_interval}, final_val_epochs={final_val_epochs})"
            )
        append_results_row(results_file, epoch, train_stats, val_stats, current_lr, train_stats["imgsz"])
        save_checkpoint(out_dir / "last.pt", model, optimizer, scheduler, epoch)
        if ema is not None:
            save_checkpoint(out_dir / "last_ema.pt", ema.ema, optimizer, scheduler, epoch)
        if val_stats.get("validated", False) and val_stats["map"] > best_map:
            best_map = val_stats["map"]
            best_epoch = epoch
            best_val_stats = copy.deepcopy(val_stats)
            best_curves = val_stats.get("curves")
            save_checkpoint(out_dir / "best.pt", eval_model, optimizer, scheduler, epoch)

    if best_val_stats is not None and best_epoch is not None:
        write_per_class_results(per_class_results_file, best_epoch, best_val_stats, config["dataset"].get("names", []))
        print_per_class_metrics(best_epoch, epochs, best_val_stats, config["dataset"].get("names", []))
    if best_curves is not None:
        plot_detection_curves(best_curves, str(out_dir), config["dataset"].get("names", []))
    plot_results(str(results_file), str(out_dir / "results.png"))
    plot_per_class_results(str(per_class_results_file), str(out_dir / "results_per_class.png"))
    print(f"saved curves to {out_dir}")


if __name__ == "__main__":
    main()
