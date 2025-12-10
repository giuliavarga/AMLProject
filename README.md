# Semantic Correspondence with DINOv2, DINOv3, and SAM

Advanced Machine Learning Project - Phase 1 Setup Complete ✅

## 🎯 Project Overview

This project implements semantic correspondence using state-of-the-art vision backbones:
- **DINOv2** - Self-supervised Vision Transformer
- **DINOv3** - Latest DINO iteration
- **SAM** - Segment Anything Model

Evaluated on **SD4Match** dataset following strict train/val/test protocol.

## 📁 Project Structure

```
AMLProject/
├── ProjectCode.ipynb           # 🔴 MAIN NOTEBOOK - Start here!
├── PHASE1_SUMMARY.md           # Setup completion summary
├── SETUP_GUIDE.md              # Comprehensive setup guide
├── QUICK_REFERENCE.md          # Quick reference card
├── PROJECT_CHECKLIST.md        # Full project checklist
│
├── checkpoints/                # Model checkpoints
│   ├── dinov3/                # ⚠️ Download DINOv3 checkpoint here
│   └── sam/                   # ✅ SAM checkpoints (auto-downloaded)
│
├── data/                       # Datasets
│   └── SD4Match/              # ⚠️ Download dataset here
│       ├── trn/               # Training split
│       ├── val/               # Validation split
│       └── test/              # Test split (final eval only!)
│
├── models/                     # Model repositories
│   ├── dinov2/                # ✅ DINOv2 repo (auto-cloned)
│   └── dinov3/                # ✅ DINOv3 repo (auto-cloned)
│
├── outputs/                    # Experiment outputs
└── utils/                      # Utility scripts
```

## 🚀 Quick Start

### 1. Open the Main Notebook
```bash
jupyter notebook ProjectCode.ipynb
```

### 2. Run All Cells
The notebook will automatically:
- ✅ Detect environment (Colab/local)
- ✅ Create directory structure
- ✅ Clone model repositories
- ✅ Download SAM checkpoint
- ✅ Load DINOv2 and SAM models
- ✅ Setup utilities and configuration

### 3. Complete Manual Steps
- ⚠️ **Download DINOv3 checkpoint** - Request access from [DINOv3 repo](https://github.com/facebookresearch/dinov3)
- ⚠️ **Download SD4Match dataset** - Follow instructions in [SD4Match repo](https://github.com/ActiveVisionLab/SD4Match)

## 📊 Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| DINOv2 ViT-B | ✅ Ready | Fully loaded and tested |
| SAM ViT-B | ✅ Ready | Fully loaded and tested |
| DINOv3 ViT-B | ⚠️ Pending | Awaiting checkpoint download |
| SD4Match Dataset | ⚠️ Pending | Need to download |
| Infrastructure | ✅ Complete | All utilities ready |
| Documentation | ✅ Complete | 4 comprehensive guides |

## 📚 Documentation

- **PHASE1_SUMMARY.md** - What's done and what's next
- **SETUP_GUIDE.md** - Complete setup instructions
- **QUICK_REFERENCE.md** - Quick reference for daily use
- **PROJECT_CHECKLIST.md** - Full project roadmap

## 🎓 Professor's Requirements

### Model Selection ✅
- Primary: **Base (ViT-B)** versions for all backbones
- Optional: Compare with different sizes if compute allows

### Dataset Protocol 🔴 IMPORTANT
- **Train** on `trn` split only
- **Validate** on `val` split for model selection
- **Test** on `test` split ONLY for final evaluation
- ⚠️ **Never train on val or test splits!**

### Evaluation
- Model selection using validation set
- Hyperparameter tuning on validation set
- Final results reported ONLY on test set
- Use SD4Match metrics

## 🛠️ Key Features Implemented

### Model Loaders
```python
# DINOv2
dinov2_model = load_dinov2_model('dinov2_vitb14', device=device)

# SAM
sam_model, sam_predictor = load_sam_model(checkpoint_path, 'vit_b', device=device)
```

### Feature Extraction
```python
# DINOv2 features
features = extract_dinov2_features(model, image)
cls_token = features['cls_token']        # (1, 768)
patch_tokens = features['patch_tokens']  # (1, 256, 768)

# SAM features
embedding = extract_sam_features(model, image)  # (1, 256, 64, 64)
```

### Utilities
```python
# Visualization
visualize_correspondence(img1, img2, pts1, pts2, matches)

# Checkpointing
save_model_checkpoint(model, optimizer, epoch, path)
load_model_checkpoint(model, path, optimizer, device)
```

## 🔗 External Resources

- [SD4Match Dataset](https://github.com/ActiveVisionLab/SD4Match) - Dataset & Metrics
- [DINOv2](https://github.com/facebookresearch/dinov2) - DINOv2 Model
- [DINOv3](https://github.com/facebookresearch/dinov3) - DINOv3 Model
- [SAM](https://github.com/facebookresearch/segment-anything) - Segment Anything
- [GeoAware-SC](https://github.com/Junyi42/geoaware-sc) - Refinement Method

## 📈 Project Phases

- [x] **Phase 1**: Setup & Infrastructure ✅ **COMPLETE**
- [ ] **Phase 2**: Dataset Integration
- [ ] **Phase 3**: Feature Extraction
- [ ] **Phase 4**: Correspondence Methods
- [ ] **Phase 5**: Training & Validation
- [ ] **Phase 6**: Final Evaluation
- [ ] **Phase 7**: Reporting & Submission

## 👥 Team Tasks

### Immediate Actions Required
1. **Someone**: Request and download DINOv3 checkpoint
2. **Someone**: Download SD4Match dataset
3. **Everyone**: Review documentation and setup

### Next Phase Tasks
- Implement dataset loader
- Create preprocessing pipeline
- Setup evaluation metrics
- Begin baseline experiments

## 💡 Tips

### For Google Colab Users
- Upload dataset to Google Drive for persistence
- Drive mounting is already configured in notebook
- Path: `/content/drive/MyDrive/AMLProject/data/`

### Performance Notes
- Base (ViT-B) models are recommended starting point
- Larger models don't always give better results
- Compare sizes if compute budget allows

## ✨ What's Included

✅ Complete setup notebook (28 cells)  
✅ Automatic environment detection  
✅ DINOv2 fully integrated  
✅ SAM fully integrated  
✅ DINOv3 repo ready (pending checkpoint)  
✅ Feature extraction utilities  
✅ Visualization tools  
✅ Checkpoint management  
✅ Configuration system  
✅ Comprehensive documentation  

## 🎉 Ready to Go!

Phase 1 setup is complete. The infrastructure is ready for your team to implement semantic correspondence methods. Good luck! 🚀

---

**Status**: Phase 1 Complete ✅  
**Last Updated**: December 10, 2025  
**Next Milestone**: Complete manual downloads + Begin Phase 2
Semantic Correspondence with Visual Foundation Models
