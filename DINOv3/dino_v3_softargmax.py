import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def select_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def parse_thresholds(raw):
    return [float(x) for x in raw.split(",") if x.strip()]


class SPairDataset(Dataset):
    def __init__(self, root, split, image_size, max_samples=None, max_pairs_per_category=None):
        self.root = Path(root)
        self.split = split
        self.img_sz = image_size
        self.max_samples = max_samples
        self.max_pairs = max_pairs_per_category
        
        split_file = self.root / "Layout" / "small" / ("trn.txt" if split == "train" else f"{split}.txt")
        if not split_file.exists():
            raise FileNotFoundError(f"Can't find: {split_file}")
        
        self._anno_cache = {}
        self._build_annotation_cache()
        
        self.pairs = []
        cat_counts = {}
        skipped = 0
        
        with open(split_file, "r") as f:
            lines = f.readlines()
            if self.max_samples:
                lines = lines[:self.max_samples]
        
        for line in lines:
            parts = line.strip().split(":")
            if len(parts) < 2:
                continue
            pair_info = parts[0].split("-")
            if len(pair_info) < 3:
                continue
            category = parts[1].strip()
            src_img = f"{pair_info[1]}.jpg"
            tgt_img = f"{pair_info[2]}.jpg"
            
            cnt = cat_counts.get(category, 0)
            if self.max_pairs is not None and cnt >= self.max_pairs:
                continue
            
            cache_key = f"{src_img}-{tgt_img}-{category}"
            if cache_key not in self._anno_cache:
                skipped += 1
                continue
            
            self.pairs.append({
                "src": src_img,
                "tgt": tgt_img,
                "category": category,
            })
            cat_counts[category] = cnt + 1
        
        print(f"Loaded {len(self.pairs)} pairs (skipped {skipped} missing annos)")
    
    def _build_annotation_cache(self):
        anno_dir = self.root / "PairAnnotation" / ("trn" if self.split == "train" else self.split)
        if not anno_dir.exists():
            anno_dir = self.root / "PairAnnotation" / "test"
        if not anno_dir.exists():
            raise FileNotFoundError(f"Missing annotation folder: {anno_dir}")
        
        for json_file in anno_dir.glob("*.json"):
            with open(json_file, "r") as f:
                data = json.load(f)
            src = data.get("src_imname", "")
            tgt = data.get("trg_imname", "") or data.get("tgt_imname", "")
            cat = data.get("category", "")
            if src and tgt and cat:
                key = f"{src}-{tgt}-{cat}"
                self._anno_cache[key] = json_file
    
    def __len__(self):
        return len(self.pairs)
    
    def _to_tensor(self, img):
        arr = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (arr - mean) / std
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        src_path = self.root / "JPEGImages" / pair["category"] / pair["src"]
        tgt_path = self.root / "JPEGImages" / pair["category"] / pair["tgt"]
        
        src_img = Image.open(src_path).convert("RGB")
        tgt_img = Image.open(tgt_path).convert("RGB")
        
        orig_src_size = src_img.size
        orig_tgt_size = tgt_img.size
        
        src_img = src_img.resize((self.img_sz, self.img_sz), Image.BILINEAR)
        tgt_img = tgt_img.resize((self.img_sz, self.img_sz), Image.BILINEAR)
        
        src_tensor = self._to_tensor(src_img)
        tgt_tensor = self._to_tensor(tgt_img)
        
        cache_key = f"{pair['src']}-{pair['tgt']}-{pair['category']}"
        anno_path = self._anno_cache[cache_key]
        with open(anno_path, "r") as f:
            anno = json.load(f)
        
        src_kps = np.array(anno["src_kps"], dtype=np.float32)
        tgt_kps = np.array(anno.get("trg_kps", anno.get("tgt_kps")), dtype=np.float32)
        
        # rescale kps
        src_kps[:, 0] = src_kps[:, 0] * self.img_sz / orig_src_size[0]
        src_kps[:, 1] = src_kps[:, 1] * self.img_sz / orig_src_size[1]
        tgt_kps[:, 0] = tgt_kps[:, 0] * self.img_sz / orig_tgt_size[0]
        tgt_kps[:, 1] = tgt_kps[:, 1] * self.img_sz / orig_tgt_size[1]
        
        valid = np.array([(kp[0] >= 0 and kp[1] >= 0) for kp in tgt_kps], dtype=bool)
        
        return {
            "src_img": src_tensor,
            "tgt_img": tgt_tensor,
            "src_kps": torch.from_numpy(src_kps),
            "tgt_kps": torch.from_numpy(tgt_kps),
            "valid": torch.from_numpy(valid),
            "category": pair["category"],
            "src_name": pair["src"],
            "tgt_name": pair["tgt"],
        }


def collate_fn(batch):
    return {
        "src_img": torch.stack([b["src_img"] for b in batch]),
        "tgt_img": torch.stack([b["tgt_img"] for b in batch]),
        "src_kps": [b["src_kps"] for b in batch],
        "tgt_kps": [b["tgt_kps"] for b in batch],
        "valid": [b["valid"] for b in batch],
        "category": [b["category"] for b in batch],
        "src_name": [b["src_name"] for b in batch],
        "tgt_name": [b["tgt_name"] for b in batch],
    }


class FeatureExtractor:
    def __init__(self, model, patch_size, device):
        self.model = model
        self.patch_size = patch_size
        self.device = device
        self.hook_out = None
    
    def _hook(self, module, _inp, output):
        self.hook_out = output[0] if isinstance(output, (list, tuple)) else output
    
    def extract(self, img, layer_idx):
        b, _, h, w = img.shape
        h_adj = (h // self.patch_size) * self.patch_size
        w_adj = (w // self.patch_size) * self.patch_size
        if h_adj != h or w_adj != w:
            img = F.interpolate(img, size=(h_adj, w_adj), mode="bilinear", align_corners=False)
        
        hook = self.model.blocks[layer_idx].register_forward_hook(self._hook)
        with torch.no_grad():
            out = self.model.forward_features(img)
        hook.remove()
        
        tokens = self.hook_out if self.hook_out is not None else out["x_norm_patchtokens"]
        self.hook_out = None
        
        if tokens.dim() == 3 and tokens.shape[1] == (h_adj // self.patch_size) * (w_adj // self.patch_size) + 1:
            tokens = tokens[:, 1:]
        
        bsz, n, dim = tokens.shape
        h_feat = h_adj // self.patch_size
        w_feat = w_adj // self.patch_size
        tokens = tokens[:, -h_feat * w_feat:]
        feats = tokens.reshape(bsz, h_feat, w_feat, dim)
        return F.normalize(feats, p=2, dim=-1)


def load_dinov3(repo_path, checkpoint, device):
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint}")
    model = torch.hub.load(str(repo_path), "dinov3_vitb16", source="local", weights=str(checkpoint))
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_finetuned_weights(model, weights_path):
    if not weights_path.exists():
        raise FileNotFoundError(f"Fine-tuned weights not found: {weights_path}")
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=False)


def softargmax_window(sim_map, window, temperature, patch):
    """Refine correspondence with soft-argmax in local window"""
    h, w = sim_map.shape
    flat_idx = torch.argmax(sim_map)
    y_peak = flat_idx // w
    x_peak = flat_idx % w
    
    # define window around peak
    radius = window // 2
    y0 = max(0, int(y_peak) - radius)
    y1 = min(h, int(y_peak) + radius + 1)
    x0 = max(0, int(x_peak) - radius)
    x1 = min(w, int(x_peak) + radius + 1)
    
    crop = sim_map[y0:y1, x0:x1]
    weights = F.softmax(crop.flatten() / temperature, dim=0)
    
    ys = torch.arange(y0, y1, device=sim_map.device, dtype=torch.float32)
    xs = torch.arange(x0, x1, device=sim_map.device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    
    refined_y = float((weights * grid_y.flatten()).sum().item())
    refined_x = float((weights * grid_x.flatten()).sum().item())
    
    return refined_x * patch + patch / 2.0, refined_y * patch + patch / 2.0


def predict_softargmax(src_feats, tgt_feats, src_kps, patch, window, temperature):
    """Predict correspondences with soft-argmax refinement"""
    b, hs, ws, dim = src_feats.shape
    _, ht, wt, _ = tgt_feats.shape
    preds = []
    
    for i in range(b):
        kps = src_kps[i]
        kps_feat = (kps / patch).long()
        kps_feat[:, 0] = kps_feat[:, 0].clamp(0, ws - 1)
        kps_feat[:, 1] = kps_feat[:, 1].clamp(0, hs - 1)
        
        current = torch.zeros_like(kps, dtype=torch.float32)
        flat_tgt = tgt_feats[i].reshape(-1, dim)
        for j in range(kps.shape[0]):
            xf = kps_feat[j, 0]
            yf = kps_feat[j, 1]
            src_vec = src_feats[i, yf, xf]
            sims = torch.matmul(src_vec, flat_tgt.T).reshape(ht, wt)
            refined_x, refined_y = softargmax_window(sims, window, temperature, patch)
            current[j, 0] = refined_x
            current[j, 1] = refined_y
        preds.append(current)
    return preds


def evaluate(model, extractor, dataloader, thresholds, image_size, feature_layer, 
             patch_size, window, softmax_temp, device):
    overall = {f"PCK@{t}": {"correct": 0, "total": 0} for t in thresholds}
    per_category = {}
    per_image = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Soft-argmax eval", leave=True):
            src_img = batch["src_img"].to(device)
            tgt_img = batch["tgt_img"].to(device)
            src_kps = [k.to(device) for k in batch["src_kps"]]
            tgt_kps = [k.to(device) for k in batch["tgt_kps"]]
            valids = [v.to(device) for v in batch["valid"]]
            
            src_feats = extractor.extract(src_img, layer_idx=feature_layer)
            tgt_feats = extractor.extract(tgt_img, layer_idx=feature_layer)
            preds = predict_softargmax(src_feats, tgt_feats, src_kps, patch_size, window, softmax_temp)
            
            for i in range(len(preds)):
                cat = batch["category"][i]
                src_name = batch["src_name"][i]
                tgt_name = batch["tgt_name"][i]
                pred = preds[i].cpu()
                gt = tgt_kps[i].cpu()
                valid = valids[i].cpu().bool()
                
                n_valid = int(valid.sum().item())
                if n_valid == 0:
                    continue
                
                correct_counts = {f"PCK@{t}": 0 for t in thresholds}
                kp_records = []
                for k in range(gt.shape[0]):
                    kp_valid = bool(valid[k].item())
                    err = float(torch.norm(pred[k] - gt[k]).item()) if kp_valid else None
                    correctness = {}
                    if kp_valid:
                        for t in thresholds:
                            ok = err <= t * image_size
                            correctness[f"PCK@{t}"] = ok
                            if ok:
                                correct_counts[f"PCK@{t}"] += 1
                    kp_records.append({
                        "index": k,
                        "valid": kp_valid,
                        "gt": [float(gt[k, 0]), float(gt[k, 1])],
                        "pred": [float(pred[k, 0]), float(pred[k, 1])],
                        "error": err,
                        "correct": correctness,
                    })
                
                img_result = {
                    "src": src_name,
                    "tgt": tgt_name,
                    "category": cat,
                    "pck": {k: (v / n_valid * 100.0) for k, v in correct_counts.items()},
                    "keypoints": kp_records,
                }
                per_image.append(img_result)
                
                if cat not in per_category:
                    per_category[cat] = {"correct": {k: 0 for k in correct_counts}, 
                                        "total": {k: 0 for k in correct_counts}, "images": 0}
                per_category[cat]["images"] += 1
                
                for t in thresholds:
                    key = f"PCK@{t}"
                    overall[key]["correct"] += correct_counts[key]
                    overall[key]["total"] += n_valid
                    per_category[cat]["correct"][key] += correct_counts[key]
                    per_category[cat]["total"][key] += n_valid
    
    overall_pck = {k: (v["correct"] / v["total"] * 100.0 if v["total"] > 0 else 0.0) for k, v in overall.items()}
    per_category_pck = {
        cat: {
            k: (data["correct"][k] / data["total"][k] * 100.0 if data["total"][k] > 0 else 0.0)
            for k in overall_pck
        } | {"images": data["images"]}
        for cat, data in per_category.items()
    }
    
    return {
        "overall": overall_pck,
        "per_category": per_category_pck,
        "per_image": per_image,
    }


def main(args):
    if args.image_size != 512:
        raise ValueError("image_size must be 512 as required by the assignment")
    
    device = select_device(args.device)
    repo_path = Path(args.repo_path)
    checkpoint = Path(args.checkpoint_path)
    weights_path = Path(args.finetuned_weights)
    dataset_path = Path(args.dataset_path)
    
    model = load_dinov3(repo_path, checkpoint, device)
    load_finetuned_weights(model, weights_path)
    extractor = FeatureExtractor(model, args.patch_size, device)
    
    dataset = SPairDataset(
        root=dataset_path,
        split="test",
        image_size=args.image_size,
        max_samples=args.max_samples,
        max_pairs_per_category=args.max_pairs_per_category,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    
    results = evaluate(model, extractor, dataloader, args.pck_thresholds, args.image_size,
                      args.feature_layer, args.patch_size, args.window_size, args.softmax_temp, device)
    
    output = {
        "phase": "softargmax",
        "config": {
            "image_size": args.image_size,
            "patch_size": args.patch_size,
            "feature_layer": args.feature_layer,
            "thresholds": args.pck_thresholds,
            "batch_size": args.batch_size,
            "max_samples": args.max_samples,
            "max_pairs_per_category": args.max_pairs_per_category,
            "device": str(device),
            "window_size": args.window_size,
            "softmax_temp": args.softmax_temp,
            "weights": str(weights_path),
        },
    } | results
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved evaluation JSON to {out_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    
    parser = argparse.ArgumentParser(description="Soft-argmax refinement for DINOv3 correspondences")
    parser.add_argument("--dataset-path", type=str, default=str(project_root / "SPair-71k"))
    parser.add_argument("--repo-path", type=str, default=str(project_root / "dinov3"))
    parser.add_argument("--checkpoint-path", type=str,
                        default=str(project_root / "checkpoints" / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"))
    parser.add_argument("--finetuned-weights", type=str, default=str(project_root / "dinov3_finetuned.pth"))
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--feature-layer", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-pairs-per-category", type=int, default=None)
    parser.add_argument("--pck-thresholds", type=parse_thresholds, default=parse_thresholds("0.05,0.1,0.15,0.2"))
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--softmax-temp", type=float, default=0.05)
    parser.add_argument("--output", type=str, default=str(project_root / "dino_v3_softargmax_results.json"))
    parser.add_argument("--device", type=str, default="auto")
    
    args = parser.parse_args()
    main(args)
    