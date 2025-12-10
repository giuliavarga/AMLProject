# Semantic Correspondence Project - Setup Guide

## Phase 1: Infrastructure Setup (Complete)

This guide documents the initial setup phase for the semantic correspondence project using DINOv2, DINOv3, and SAM backbones.

## Project Structure

```
AMLProject/
├── ProjectCode.ipynb       # Main notebook with all setup code
├── checkpoints/            # Model checkpoints
│   ├── dinov3/            # DINOv3 checkpoints (to be downloaded)
│   └── sam/               # SAM checkpoints (auto-downloaded)
├── data/                   # Dataset directory
│   └── SD4Match/          # SD4Match dataset (to be downloaded)
│       ├── trn/           # Training split
│       ├── val/           # Validation split
│       └── test/          # Test split
├── models/                 # Model repositories
│   ├── dinov2/            # DINOv2 official repo (auto-cloned)
│   └── dinov3/            # DINOv3 official repo (auto-cloned)
├── outputs/                # Experiment outputs
└── utils/                  # Utility scripts
```

## Setup Checklist

### ✅ Completed Automatically
- [x] Environment configuration (Colab/local detection)
- [x] Directory structure creation
- [x] DINOv2 repository cloning
- [x] DINOv2 ViT-B model loading
- [x] DINOv3 repository cloning
- [x] SAM installation
- [x] SAM ViT-B checkpoint download
- [x] SAM model loading
- [x] Utility functions for feature extraction
- [x] Visualization utilities
- [x] Configuration management

### ⚠️ Manual Steps Required

#### 1. DINOv3 Checkpoint Access
**Status**: Repository cloned, checkpoint needs downloading

**Steps**:
1. Request access to DINOv3 checkpoints from Facebook Research
2. Follow instructions in the [DINOv3 repository](https://github.com/facebookresearch/dinov3)
3. Download the **ViT-B (Base)** checkpoint
4. Place it in: `checkpoints/dinov3/dinov3_vitb14_pretrain.pth`
5. Update the loading code in the notebook once checkpoint structure is known

#### 2. SD4Match Dataset Download
**Status**: Repository cloned, dataset needs downloading

**Steps**:
1. Visit the [SD4Match repository](https://github.com/ActiveVisionLab/SD4Match)
2. Follow dataset download instructions
3. Place dataset in: `data/SD4Match/`
4. Ensure you have these splits:
   - `trn/` - Training split
   - `val/` - Validation split
   - `test/` - Test split

**For Google Colab Users**:
- Upload the dataset to Google Drive for persistence
- Mount Drive in the notebook (already configured)
- Path will be: `/content/drive/MyDrive/AMLProject/data/SD4Match/`

## Model Information

### DINOv2 (✅ Ready)
- **Model**: ViT-B/14 (Base with 14x14 patches)
- **Source**: Loaded via `torch.hub` from official repo
- **Status**: ✅ Loaded and ready to use
- **Features**: Access to CLS token and patch tokens
- **Function**: `extract_dinov2_features(model, image)`

### DINOv3 (⚠️ Pending)
- **Model**: ViT-B/14 (Base with 14x14 patches)
- **Source**: Official checkpoint (access required)
- **Status**: ⚠️ Awaiting checkpoint download
- **Next Step**: Request access and download checkpoint

### SAM (✅ Ready)
- **Model**: ViT-B (Base)
- **Source**: Downloaded from Facebook Research
- **Status**: ✅ Loaded and ready to use
- **Features**: Image encoder embeddings
- **Function**: `extract_sam_features(model, image)`

### Optional: Different Backbone Sizes
You can experiment with different sizes for comparison:

**DINOv2**:
- `dinov2_vits14` - Small
- `dinov2_vitb14` - Base (current)
- `dinov2_vitl14` - Large
- `dinov2_vitg14` - Giant

**SAM**:
- `vit_b` - Base (current)
- `vit_l` - Large
- `vit_h` - Huge

## Professor's Guidelines

### Backbone Selection
- ✅ Use **Base (ViT-B)** versions as primary choice
- Consider comparing sizes if compute budget allows
- Larger models may not always give proportional improvements

### Dataset Usage Protocol
- **Training**: Use `trn` split
- **Validation**: Use `val` split for model selection/tuning
- **Testing**: Use `test` split ONLY for final evaluation and reporting
- ⚠️ **Never train on val or test splits**

### Model Access
- DINOv2: Use official repository (not just Hugging Face) for internal component access
- DINOv3: Request official checkpoint access
- SAM: Public checkpoints available directly

## Running the Notebook

### Local Execution
```bash
cd "/Users/giuliavarga/Desktop/2. AML/Project/AMLProject"
jupyter notebook ProjectCode.ipynb
```

### Google Colab
1. Upload notebook to Google Drive
2. Open with Google Colab
3. Mount Drive for data access
4. Run all cells sequentially

## Utilities Provided

### Feature Extraction
- `extract_dinov2_features(model, image)` - Extract DINOv2 features
- `extract_sam_features(model, image)` - Extract SAM features

### Visualization
- `visualize_correspondence(img1, img2, pts1, pts2, matches)` - Show correspondence

### Checkpointing
- `save_model_checkpoint(model, optimizer, epoch, path)` - Save training state
- `load_model_checkpoint(model, path, optimizer)` - Resume training

### Configuration
- `ProjectConfig` class - Central configuration management

## Next Steps for Team

1. **Complete Manual Steps**:
   - Download DINOv3 checkpoint
   - Download SD4Match dataset

2. **Verify Setup**:
   - Run test inference to verify all models work
   - Check dataset loading

3. **Implement Correspondence**:
   - Feature extraction pipeline
   - Matching algorithms
   - Refinement methods (GeoAware-SC)

4. **Evaluation**:
   - Implement SD4Match metrics
   - Run experiments on train/val
   - Final evaluation on test split only

5. **Documentation**:
   - Record experimental results
   - Compare backbone performances
   - Analyze size vs. performance trade-offs

## Additional Resources

- **SD4Match**: https://github.com/ActiveVisionLab/SD4Match
- **DINOv2**: https://github.com/facebookresearch/dinov2
- **DINOv3**: https://github.com/facebookresearch/dinov3
- **SAM**: https://github.com/facebookresearch/segment-anything
- **GeoAware-SC**: https://github.com/Junyi42/geoaware-sc

## Support

If you encounter issues:
1. Check that all dependencies are installed
2. Verify CUDA availability for GPU acceleration
3. Ensure paths are correctly set for your environment
4. Check that downloaded files are in correct locations

## Contact

For team collaboration and questions, refer to project documentation and professor's guidelines.
