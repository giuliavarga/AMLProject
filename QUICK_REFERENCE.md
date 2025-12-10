# Quick Reference - Semantic Correspondence Project

## 🚀 Quick Start

### Run the Setup Notebook
1. Open `ProjectCode.ipynb`
2. Run all cells in order
3. Models will auto-download (except DINOv3)

## ✅ Ready to Use

### DINOv2 ViT-B
```python
# Load model
dinov2_model = load_dinov2_model('dinov2_vitb14', device=device)

# Extract features
features = extract_dinov2_features(dinov2_model, image)
cls_token = features['cls_token']        # Shape: (1, 768)
patch_tokens = features['patch_tokens']  # Shape: (1, 256, 768)
```

### SAM ViT-B
```python
# Load model
sam_model, sam_predictor = load_sam_model(sam_checkpoint_path, 'vit_b', device=device)

# Extract features
embedding = extract_sam_features(sam_model, image)
# Shape: (1, 256, 64, 64)
```

## ⚠️ TODO Items

### 1. DINOv3 Checkpoint
- [ ] Request access: https://github.com/facebookresearch/dinov3
- [ ] Download ViT-B checkpoint
- [ ] Place in: `checkpoints/dinov3/dinov3_vitb14_pretrain.pth`
- [ ] Update loader code after understanding checkpoint format

### 2. SD4Match Dataset
- [ ] Download from: https://github.com/ActiveVisionLab/SD4Match
- [ ] Place in: `data/SD4Match/`
- [ ] Verify splits: `trn/`, `val/`, `test/`

## 📋 Dataset Usage Rules

| Split | Purpose | Usage |
|-------|---------|-------|
| `trn` | Training | Train models |
| `val` | Validation | Model selection, hyperparameter tuning |
| `test` | Testing | **FINAL EVALUATION ONLY** |

⚠️ **NEVER train on val or test!**

## 🎯 Model Versions

### Recommended (Base)
- DINOv2: `dinov2_vitb14`
- DINOv3: `vitb14`
- SAM: `vit_b`

### Optional Comparison
| Size | DINOv2 | SAM |
|------|--------|-----|
| Small | `dinov2_vits14` | - |
| Base | `dinov2_vitb14` ✅ | `vit_b` ✅ |
| Large | `dinov2_vitl14` | `vit_l` |
| Huge/Giant | `dinov2_vitg14` | `vit_h` |

## 🛠️ Key Functions

### Feature Extraction
```python
# DINOv2
features = extract_dinov2_features(model, image)

# SAM
embedding = extract_sam_features(model, image)
```

### Visualization
```python
fig = visualize_correspondence(img1, img2, pts1, pts2, matches)
plt.show()
```

### Checkpointing
```python
# Save
save_model_checkpoint(model, optimizer, epoch, path)

# Load
checkpoint = load_model_checkpoint(model, path, optimizer, device)
```

## 📁 Directory Structure

```
AMLProject/
├── ProjectCode.ipynb       # Main notebook
├── checkpoints/
│   ├── dinov3/            # ⚠️ DINOv3 checkpoints (manual)
│   └── sam/               # ✅ SAM checkpoints (auto)
├── data/
│   └── SD4Match/          # ⚠️ Dataset (manual)
├── models/
│   ├── dinov2/            # ✅ Cloned automatically
│   └── dinov3/            # ✅ Cloned automatically
├── outputs/                # Your results
└── utils/                  # Helper scripts
```

## 🔗 Important Links

- [SD4Match Dataset](https://github.com/ActiveVisionLab/SD4Match)
- [DINOv2 Repo](https://github.com/facebookresearch/dinov2)
- [DINOv3 Repo](https://github.com/facebookresearch/dinov3)
- [SAM Repo](https://github.com/facebookresearch/segment-anything)
- [GeoAware-SC (Refinement)](https://github.com/Junyi42/geoaware-sc)

## 💡 Tips

### For Colab Users
- Upload dataset to Google Drive
- Mount Drive: already configured in notebook
- Path: `/content/drive/MyDrive/AMLProject/data/`

### Performance Notes
- Larger backbones ≠ always better
- Base models are good starting point
- Compare sizes if compute budget allows
- Focus on proper evaluation protocol

## 🎓 Professor's Key Points

1. **Use official repos** (not just Hugging Face) for internal access
2. **Base (ViT-B)** is the recommended size
3. **Test split** is for final evaluation ONLY
4. Size comparison is optional but interesting
5. Gains from larger models aren't always consistent

## 🔧 Troubleshooting

### CUDA Out of Memory
- Use smaller batch size
- Switch to smaller backbone
- Enable gradient checkpointing

### Import Errors
- Check sys.path includes model directories
- Reinstall packages: `!pip install -r requirements.txt`

### Checkpoint Not Found
- Verify download paths
- Check file permissions
- Re-run download cells

## 📊 Next Phase Preview

After setup is complete:
1. Implement feature extraction pipeline
2. Develop matching algorithms
3. Integrate GeoAware-SC refinement
4. Implement SD4Match evaluation metrics
5. Run experiments (train → val → test)
6. Analyze results and compare backbones

---

**Status**: Phase 1 Complete ✅  
**Ready for**: Team implementation of correspondence methods  
**Pending**: DINOv3 checkpoint + SD4Match dataset downloads
