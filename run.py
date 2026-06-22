import argparse
import json
import os
import random


pre_parser = argparse.ArgumentParser(add_help=False)
pre_parser.add_argument("--gpu_id", default="0", type=str)
pre_args, _ = pre_parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = pre_args.gpu_id

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import DA_DatasetFromFolder
from data_utils import LoadDatasetFromFolder
from data_utils import calMetric_iou
from losses import WeightedBCEDiceLoss
from losses import normalize_binary_target
from model import ReMambaNet


def seed_torch(seed=2026):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def extract_logits(output):
    if isinstance(output, dict):
        return output["change_pred"]
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def checkpoint_dir(args):
    if args.output_dir:
        return args.output_dir
    if args.run_name:
        return os.path.join(args.save_dir, args.run_name, str(args.epochs))
    return os.path.join(args.save_dir, args.dataset, str(args.epochs))


def load_checkpoint(model, model_path, device):
    if not model_path:
        return
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Initial checkpoint not found: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    cleaned = {key[7:] if key.startswith("module.") else key: value for key, value in checkpoint.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[Init Load Warning] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[Init Load Warning] Unexpected keys: {len(unexpected)}")
    print(f"Loaded initial checkpoint: {model_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train ReMamba for remote sensing change detection.")
    parser.add_argument("--gpu_id", default="0", type=str)
    parser.add_argument("--dataset", type=str, default="BCDD")
    parser.add_argument("--batch_size", "--batchsize", dest="batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_remamba")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--init_model_path", type=str, default="")
    parser.add_argument("--init_save_dir", type=str, default="./checkpoints_remamba")
    parser.add_argument("--init_checkpoint_dataset", "--source_dataset", dest="init_checkpoint_dataset", type=str, default="")
    parser.add_argument("--init_checkpoint_epochs", type=str, default="200")
    parser.add_argument("--init_checkpoint_name", type=str, default="netCD_epoch_best.pth")

    parser.add_argument("--hr1_train", default="./rs_cd/", type=str)
    parser.add_argument("--hr2_train", default="./rs_cd/", type=str)
    parser.add_argument("--lab_train", default="./rs_cd/", type=str)
    parser.add_argument("--hr1_val", default="./rs_cd/", type=str)
    parser.add_argument("--hr2_val", default="./rs_cd/", type=str)
    parser.add_argument("--lab_val", default="./rs_cd/", type=str)
    parser.add_argument("--suffix", nargs="+", default=[".png", ".jpg", ".tif", ".tiff"])

    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scan_backend", choices=["fast", "torch"], default="fast")
    parser.add_argument("--ssm_d_state", type=int, default=8)
    parser.add_argument("--ssm_ratio", type=float, default=1.0)
    parser.add_argument("--lambda_reweight", type=float, default=1.0)

    parser.add_argument("--pos_weight", type=float, default=2.0)
    parser.add_argument("--dynamic_pos_weight", action="store_true")
    parser.add_argument("--dice_weight", "--eta", dest="dice_weight", type=float, default=0.5)
    parser.add_argument("--early_stop_patience", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    seed_torch(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using gpu_id={args.gpu_id}, device={device}")
    print(f"ReMamba scan backend: {args.scan_backend}")

    train_paths = [
        os.path.join(args.hr1_train, args.dataset, "train/A"),
        os.path.join(args.hr2_train, args.dataset, "train/B"),
        os.path.join(args.lab_train, args.dataset, "train/label"),
    ]
    val_paths = [
        os.path.join(args.hr1_val, args.dataset, "val/A"),
        os.path.join(args.hr2_val, args.dataset, "val/B"),
        os.path.join(args.lab_val, args.dataset, "val/label"),
    ]

    train_set = DA_DatasetFromFolder(
        *train_paths,
        crop=True,
        crop_size=args.crop_size,
        augment=not args.no_augment,
        suffixes=args.suffix,
    )
    val_set = LoadDatasetFromFolder(args, *val_paths)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = ReMambaNet(
        in_channels=3,
        pretrained=args.pretrained,
        scan_backend=args.scan_backend,
        ssm_d_state=args.ssm_d_state,
        ssm_ratio=args.ssm_ratio,
        lambda_reweight=args.lambda_reweight,
    ).to(device)

    if not args.init_model_path and args.init_checkpoint_dataset:
        args.init_model_path = os.path.join(
            args.init_save_dir,
            args.init_checkpoint_dataset,
            str(args.init_checkpoint_epochs),
            args.init_checkpoint_name,
        )
    if args.init_model_path:
        load_checkpoint(model, args.init_model_path, device)

    criterion = WeightedBCEDiceLoss(
        pos_weight=args.pos_weight,
        dice_weight=args.dice_weight,
        dynamic_pos_weight=args.dynamic_pos_weight,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    ckpt_dir = checkpoint_dir(args)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    best_iou = 0.0
    epochs_without_improvement = 0
    best_model_path = os.path.join(ckpt_dir, "netCD_epoch_best.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for img1, img2, label, *_ in train_bar:
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)
            label = normalize_binary_target(label.to(device, non_blocking=True))

            optimizer.zero_grad(set_to_none=True)
            output = model(img1, img2)
            loss = criterion(output, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        inter_sum = 0.0
        union_sum = 0.0
        with torch.no_grad():
            for img1, img2, label, *_ in tqdm(val_loader, desc="Validating"):
                img1 = img1.to(device, non_blocking=True)
                img2 = img2.to(device, non_blocking=True)
                label = normalize_binary_target(label.to(device, non_blocking=True))

                logits = extract_logits(model(img1, img2))
                pred = (torch.sigmoid(logits) > 0.5).float()

                for i in range(img1.size(0)):
                    intr, unn = calMetric_iou(label[i, 0].cpu().numpy(), pred[i, 0].cpu().numpy())
                    inter_sum += intr
                    union_sum += unn

        val_iou = inter_sum / (union_sum + 1e-6)
        print(f"Epoch {epoch} - Val IoU: {val_iou:.4f}, Best IoU: {best_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"--> Saved Best Model to {best_model_path}")
        else:
            epochs_without_improvement += 1

        scheduler.step()

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(
                f"Early stop at epoch {epoch}: no Val IoU improvement for "
                f"{epochs_without_improvement} epochs. Best IoU: {best_iou:.4f}"
            )
            break


if __name__ == "__main__":
    main()
