# Semantic Correspondence Project - Setup Guide

## Current Repo Snapshot (Dec 11, 2025)
- Notebooks to run: DINOv2_Correspondence.ipynb, DINOv3_Correspondence.ipynb, SAM_Correspondence.ipynb (no ProjectCode.ipynb).
- Checkpoints: checkpoints/sam/sam_vit_b_01ec64.pth is present; no DINOv3 checkpoint yet.
- Data: data/SD4Match/pf-pascal_image_pairs.zip is downloaded but not extracted; data/PF-dataset-PASCAL and data/SPair-71k are currently empty.
- Outputs: outputs/sam exists but is empty.
- Git LFS: repository expects Git LFS; install via `brew install git-lfs && git lfs install` before pushing large files.
- Note: Cell references in this guide reflect the original consolidated notebook; use the equivalent sections in the backbone-specific notebooks.

## 🚀 Quick Setup (Choose Your Platform)

### macOS (Intel or Apple Silicon)
```bash
# 1. Create conda environment
conda create -n aml_project python=3.11 -y
conda activate aml_project

# 2. Install PyTorch (via conda for binary compatibility)
conda install pytorch torchvision torchaudio -c pytorch -y

# 3. Install dependencies
pip install opencv-python matplotlib numpy scipy tqdm einops pillow requests timm pandas

# 4. Navigate and open a notebook (choose backbone)
cd /path/to/AMLProject
jupyter notebook DINOv2_Correspondence.ipynb  # or DINOv3_Correspondence.ipynb, SAM_Correspondence.ipynb
```

### Linux (with CUDA GPU)
```bash
# 1. Create conda environment
conda create -n aml_project python=3.11 -y
conda activate aml_project

# 2. Install PyTorch with CUDA support
conda install pytorch::pytorch pytorch::torchvision pytorch::torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 3. Install dependencies
pip install opencv-python matplotlib numpy scipy tqdm einops pillow requests timm pandas

# 4. Navigate and open a notebook (choose backbone)
cd /path/to/AMLProject
jupyter notebook DINOv2_Correspondence.ipynb  # or DINOv3_Correspondence.ipynb, SAM_Correspondence.ipynb
```

### Windows (CPU or CUDA)
```bash
# 1. Create conda environment
conda create -n aml_project python=3.11 -y
conda activate aml_project

# 2. Install PyTorch
# For CPU or CUDA: conda will auto-detect
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 3. Install dependencies
pip install opencv-python matplotlib numpy scipy tqdm einops pillow requests timm pandas

# 4. Open notebook with Jupyter or VS Code
cd path\to\AMLProject
jupyter notebook DINOv2_Correspondence.ipynb  # or DINOv3_Correspondence.ipynb, SAM_Correspondence.ipynb
```

### Google Colab
```python
# Run this in first notebook cell:
from google.colab import drive
drive.mount('/content/drive')

# Set paths:
PROJECT_ROOT = '/content/drive/MyDrive/AMLProject'
import os
os.makedirs(PROJECT_ROOT, exist_ok=True)

# Run all notebook cells sequentially
# Everything else is automatic!
```

---

## 📋 Setup Checklist

### ✅ Completed Automatically (in notebook)
- [x] Environment configuration (OS/Colab detection)
- [x] Directory structure creation
- [x] Package installation
- [x] DINOv2 repository cloning
- [x] DINOv2 ViT-B model loading
- [x] DINOv3 repository cloning
- [x] SAM installation and checkpoint download
- [x] SAM ViT-B model loading
- [x] Utility functions for feature extraction
- [x] Visualization utilities
- [x] Configuration management
- [x] Test inference on loaded models

### ⚠️ Manual Steps Required

#### 1. DINOv3 Checkpoint Access
**Current Status**: Repository cloned ✅, checkpoint needed ⚠️

**What to do**:
1. Visit [DINOv3 GitHub Repository](https://github.com/facebookresearch/dinov3)
2. Check if checkpoints are available for download
3. If access is restricted, request access from Facebook Research
4. Download the **ViT-B checkpoint** (usually named `dinov3_vitb14_pretrain.pth`)
5. Save to: `checkpoints/dinov3/dinov3_vitb14_pretrain.pth`
6. Once downloaded, uncomment the DINOv3 loading code in cell 17

**Time estimate**: 2-24 hours (depends on checkpoint availability)

#### 2. SD4Match Dataset Download
**Current Status**: pf-pascal_image_pairs.zip downloaded ⚠️ (not extracted); other splits missing

**What to do**:
1. Visit [SD4Match Repository](https://github.com/ActiveVisionLab/SD4Match)
2. Follow their dataset download instructions
3. Already downloaded: `data/SD4Match/pf-pascal_image_pairs.zip` (extract into `data/SD4Match/pf-pascal/`)
4. Still needed: pf-willow and spair-71k splits (currently absent)
5. After extraction, verify structure:
    ```
    data/SD4Match/
    ├── pf-pascal/
    │   ├── pf-pascal_image_pairs/
    │   ├── PF-dataset-PASCAL/
    │   └── test_pairs.csv
    ├── pf-willow/
    │   ├── test_pairs.csv
    │   └── PF-dataset/
    └── spair-71k/
         └── SPair-71k/
    ```

**For Google Colab Users**:
- Download dataset on your computer first
- Upload to Google Drive: `MyDrive/AMLProject/data/SD4Match/`
- Notebook will auto-detect and use it

**Time estimate**: 2-6 hours (depends on internet speed and which benchmarks)

---

## 📁 Project Structure (current)

```
AMLProject/
├── DINOv2_Correspondence.ipynb       # DINOv2 pipeline notebook
├── DINOv3_Correspondence.ipynb       # DINOv3 pipeline notebook
├── SAM_Correspondence.ipynb          # SAM pipeline notebook
├── README.md                         # Project overview
├── SETUP_GUIDE.md                    # This file
├── PROJECT_CHECKLIST.md              # Roadmap and status
├── README_BACKBONES.md               # Backbone quickstart
├── BACKBONE_COMPARISON_REPORT.md     # Detailed backbone analysis
│
├── checkpoints/
│   └── sam/
│       └── sam_vit_b_01ec64.pth      # Present
│   # (no dinov3 checkpoint yet)
│
├── data/
│   ├── SD4Match/
│   │   └── pf-pascal_image_pairs.zip  # Downloaded, not extracted
│   ├── PF-dataset-PASCAL/             # Empty placeholder
│   └── SPair-71k/                     # Empty placeholder
│
├── models/
│   ├── dinov2/
│   └── dinov3/
│
├── outputs/
│   └── sam/                           # Currently empty
│
└── utils/                             # Planned utilities
```

Note: Git LFS is enabled for checkpoints/sam/. Install LFS before pushing checkpoints.

---

## 🔧 Model Information

### DINOv2 ViT-B/14 (✅ Ready)
| Property | Details |
|----------|---------|
| Status | ✅ Fully loaded and tested |
| Loaded in | Cell 12 |
| Feature dimension | 768 |
| Patch size | 14×14 |
| Input size | 224×224 |
| Output shape | (1, 256, 768) for patches + (1, 768) for CLS token |
| Source | `torch.hub` (via hubconf.py) |
| Usage | `extract_dinov2_features(model, image)` |

### SAM ViT-B (✅ Ready)
| Property | Details |
|----------|---------|
| Status | ✅ Fully loaded and tested |
| Loaded in | Cell 25 |
| Feature dimension | 256 |
| Patch size | 16×16 |
| Input size | 1024×1024 |
| Output shape | (1, 256, 64, 64) |
| Checkpoint | Auto-downloaded in cell 24 |
| Source | Facebook Research official |
| Usage | `extract_sam_features(model, image)` |

### DINOv3 ViT-B/14 (⚠️ Pending)
| Property | Details |
|----------|---------|
| Status | ⚠️ Checkpoint needed |
| To be loaded in | Cell 17 |
| Feature dimension | 768 |
| Patch size | 14×14 |
| Input size | 224×224 |
| Output shape | (1, 256, 768) for patches + (1, 768) for CLS token |
| Source | Facebook Research (request access) |
| Usage | Same as DINOv2 once loaded |

### Optional: Other Model Sizes
```python
# DINOv2 variants (can be loaded same way)
'dinov2_vits14'   # Small (11M params)
'dinov2_vitb14'   # Base (86M params) ← RECOMMENDED
'dinov2_vitl14'   # Large (300M params)
'dinov2_vitg14'   # Giant (1.1B params)

# SAM variants
sam_vit_b_01ec64  # Base (91M params) ← RECOMMENDED
sam_vit_l_0b3195  # Large (308M params)
sam_vit_h_4b8939  # Huge (632M params)
```

---

## 🎓 Professor's Guidelines

### Backbone Selection
- **Primary choice**: Base (ViT-B) for all models
  - Good balance of performance and efficiency
  - Fastest inference
  - Sufficient capacity for this task
- **Optional comparisons**: Larger sizes if you have GPU/compute budget
- **Report**: Always report base model results first, comparisons secondary

### Dataset Usage - STRICT PROTOCOL
```
Training split ('trn'):   Use only if fine-tuning models
Validation split ('val'): Use for hyperparameter tuning and model selection
Test split ('test'):      Use ONLY for final results (NEVER tune on this!)
```

### Evaluation Metrics
- **Primary metric**: PCK (Percentage of Correct Keypoints)
  - PCK@0.05 (strict)
  - PCK@0.10 (moderate)
  - PCK@0.15 (loose)
- **Secondary**: Per-category performance analysis
  - Different object types
  - Viewpoint changes
  - Scale variations

### Final Reporting Requirements
- Report results on `test` split ONLY
- Include all three PCK thresholds
- Compare backbones (DINOv2 vs SAM vs DINOv3)
- Document all hyperparameters used
- Ensure reproducibility (fixed seeds, documented environment)

---

## 🔍 Verification Steps

After running notebook cells 1-33, verify everything works:

### Check 1: Test DINOv2 Loading
```python
# In a notebook cell, run:
import torch
if dinov2_model is not None:
    test_img = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        feat = dinov2_model(test_img)
    print(f"✅ DINOv2 feature shape: {feat.shape}")
else:
    print("❌ DINOv2 model not loaded")
```

### Check 2: Test SAM Loading
```python
# In a notebook cell, run:
if sam_model is not None:
    test_img = torch.randn(1, 3, 1024, 1024).to(device)
    with torch.no_grad():
        feat = sam_model.image_encoder(test_img)
    print(f"✅ SAM feature shape: {feat.shape}")
else:
    print("❌ SAM model not loaded")
```

### Check 3: Test Dataset Loading
```python
# Once dataset is downloaded, uncomment Example 1 in notebook:
# pf_pascal = PFPascalDataset(...)
# sample = pf_pascal[0]
# print(f"✅ Sample keys: {sample.keys()}")
```

### Check 4: Test Evaluation Pipeline
```python
# Once everything is ready, uncomment Example 2:
# results = evaluate_correspondence(...)
# print(f"✅ PCK@0.10 = {results['mean']['pck_010']:.2%}")
```

---

## 🐛 Troubleshooting

### Issue: Import errors (torch, torchvision, etc.)
**Solution**:
```bash
# Reinstall matching versions via conda
conda install pytorch torchvision -c pytorch -y
# Then restart your kernel/notebook
```

### Issue: CUDA out of memory
**Solution**:
```python
# Reduce batch size or image resolution
# Or run on CPU (slower but works)
device = torch.device('cpu')
```

### Issue: Checkpoints not found
**Solution**:
```python
# Verify paths are correct
import os
print(f"CHECKPOINT_DIR exists: {os.path.exists(CHECKPOINT_DIR)}")
print(f"SAM checkpoint exists: {os.path.exists(sam_checkpoint_path)}")
```

### Issue: Dataset not loading
**Solution**:
```python
# Verify dataset structure
import os
print(os.listdir(os.path.join(DATA_ROOT, 'SD4Match')))
# Should show: pf-pascal, pf-willow, spair-71k
```

### Issue: Kernel needs restart
**Reason**: After torch/torchvision fixes or package installations
**Solution**:
```
In Jupyter/VS Code: Kernel → Restart Kernel
Then re-run cells from the beginning
```

---

## 📚 Next Steps

1. **Complete this setup** (install packages, clone repos, load models)
2. **Download DINOv3 checkpoint** (if available/accessible)
3. **Download SD4Match dataset** (at least PF-Pascal for quick testing)
4. **Run Example 1** (verify dataset loading works)
5. **Run Example 2** (verify DINOv2 baseline evaluation works)
6. **Proceed to Phase 2** (implement improvements, compare methods)

---

## 📞 Support & Questions

If you encounter issues not covered here:
1. Check the README.md for more context
2. Review QUICK_REFERENCE.md for common commands
3. Check PROJECT_CHECKLIST.md for timeline and milestones
4. Post questions to team channel or professor
5. Refer to external repositories' documentation


