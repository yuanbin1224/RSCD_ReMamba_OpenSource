import argparse
import csv
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import TestDatasetFromFolder
from data_utils import calMetric_iou
from data_utils import calMetric_somemetric
from model import ReMambaNet


def parse_args():
    parser = argparse.ArgumentParser(description="Inference and evaluation for ReMamba.")
    parser.add_argument("--dataset_name", "--dataset", dest="dataset_name", default="BCDD", type=str)
    parser.add_argument("--target_dataset", default="", type=str)
    parser.add_argument("--checkpoint_dataset", "--source_dataset", dest="checkpoint_dataset", default="", type=str)
    parser.add_argument("--gpu_id", default="0", type=str)
    parser.add_argument("--epochs", default="200", type=str)
    parser.add_argument("--save_dir", default="./checkpoints_remamba", type=str)
    parser.add_argument("--checkpoint_name", default="netCD_epoch_best.pth", type=str)
    parser.add_argument("--model_path", default="", type=str)
    parser.add_argument("--result_dir", default="./results_remamba", type=str)
    parser.add_argument("--result_subdir", default="", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--data_root", default="/data5/rs_cd", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--hr1_dir", default="", type=str)
    parser.add_argument("--hr2_dir", default="", type=str)
    parser.add_argument("--label_dir", default="", type=str)
    parser.add_argument("--suffix", nargs="+", default=[".png", ".jpg", ".tif", ".tiff"])
    parser.add_argument("--threshold", default=0.5, type=float)

    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scan_backend", choices=["fast", "torch"], default="fast")
    parser.add_argument("--ssm_d_state", type=int, default=8)
    parser.add_argument("--ssm_ratio", type=float, default=1.0)
    parser.add_argument("--lambda_reweight", type=float, default=1.0)

    parser.add_argument("--save_outputs", action="store_true")
    return parser.parse_args()


def load_checkpoint(model, model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")
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
        print(f"[Load Warning] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[Load Warning] Unexpected keys: {len(unexpected)}")


def output_dir(args, checkpoint_dataset, target_dataset):
    if args.result_subdir:
        name = args.result_subdir
    elif checkpoint_dataset != target_dataset:
        name = f"{checkpoint_dataset}_to_{target_dataset}"
    else:
        name = target_dataset
    return os.path.join(args.result_dir, name)


def save_mask(path, pred):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, pred.astype(np.uint8) * 255)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_dataset = args.target_dataset or args.dataset_name
    checkpoint_dataset = args.checkpoint_dataset or target_dataset

    if not args.hr1_dir:
        args.hr1_dir = os.path.join(args.data_root, target_dataset, args.split, "A")
    if not args.hr2_dir:
        args.hr2_dir = os.path.join(args.data_root, target_dataset, args.split, "B")
    if not args.label_dir:
        args.label_dir = os.path.join(args.data_root, target_dataset, args.split, "label")
    if not args.model_path:
        args.model_path = os.path.join(args.save_dir, checkpoint_dataset, args.epochs, args.checkpoint_name)

    save_results = output_dir(args, checkpoint_dataset, target_dataset)
    print(f"Loading ReMamba model from {args.model_path}...")
    print(f"Source checkpoint dataset: {checkpoint_dataset}")
    print(f"Target evaluation dataset: {target_dataset}")
    print(f"Evaluation split: {args.split}")

    model = ReMambaNet(
        pretrained=args.pretrained,
        scan_backend=args.scan_backend,
        ssm_d_state=args.ssm_d_state,
        ssm_ratio=args.ssm_ratio,
        lambda_reweight=args.lambda_reweight,
    ).to(device)
    load_checkpoint(model, args.model_path, device)
    model.eval()

    test_set = TestDatasetFromFolder(args, args.hr1_dir, args.hr2_dir, args.label_dir)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    inter_sum = 0.0
    union_sum = 0.0
    tp_all, tn_all, fp_all, fn_all = 0, 0, 0, 0

    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for img1, img2, label, image_names in test_bar:
            img1 = img1.to(device, dtype=torch.float)
            img2 = img2.to(device, dtype=torch.float)

            output = model(img1, img2, return_features=False)
            logits = output["change_pred"] if isinstance(output, dict) else output
            prob_tensor = torch.sigmoid(logits)
            pred_tensor = (prob_tensor > args.threshold).float()

            for sample_idx in range(img1.size(0)):
                pred = pred_tensor[sample_idx, 0].detach().cpu().numpy().astype(np.uint8)
                gt = (label[sample_idx].squeeze().cpu().numpy() > 0).astype(np.uint8)

                intr, unn = calMetric_iou(gt, pred)
                inter_sum += intr
                union_sum += unn
                tp, tn, fp, fn = calMetric_somemetric(pred, gt)
                tp_all += tp
                tn_all += tn
                fp_all += fp
                fn_all += fn

                if args.save_outputs:
                    image_stem = os.path.splitext(os.path.basename(image_names[sample_idx]))[0]
                    save_mask(os.path.join(save_results, "binary", f"{image_stem}.png"), pred)

            current_iou = inter_sum / (union_sum + 1e-6)
            test_bar.set_description(desc=f"Current IoU: {current_iou:.4f}")

    precision = tp_all / (tp_all + fp_all + 1e-6)
    recall = tp_all / (tp_all + fn_all + 1e-6)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-6)
    iou = inter_sum / (union_sum + 1e-6)
    oa = (tp_all + tn_all) / (tp_all + tn_all + fp_all + fn_all + 1e-6)

    if args.save_outputs:
        os.makedirs(save_results, exist_ok=True)
        summary_path = os.path.join(save_results, "metrics.csv")
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "source_dataset",
                    "target_dataset",
                    "split",
                    "checkpoint_epoch",
                    "threshold",
                    "iou",
                    "f1",
                    "precision",
                    "recall",
                    "oa",
                    "tp",
                    "tn",
                    "fp",
                    "fn",
                    "model_path",
                ]
            )
            writer.writerow(
                [
                    checkpoint_dataset,
                    target_dataset,
                    args.split,
                    args.epochs,
                    args.threshold,
                    iou,
                    f1,
                    precision,
                    recall,
                    oa,
                    tp_all,
                    tn_all,
                    fp_all,
                    fn_all,
                    args.model_path,
                ]
            )

    print("\n" + "=" * 50)
    print(f"FINAL TEST RESULTS ({checkpoint_dataset} -> {target_dataset}):")
    print(f"IoU: {iou:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"OA: {oa:.4f}")
    if args.save_outputs:
        print(f"Results saved to: {save_results}")
    print("=" * 50)


if __name__ == "__main__":
    main()
