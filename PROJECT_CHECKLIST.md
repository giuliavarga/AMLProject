# Project Checklist - Semantic Correspondence

## Phase 1: Setup & Infrastructure ✅

### Environment Setup
- [x] Configure paths (local/Colab detection)
- [x] Create directory structure
- [x] Install base dependencies (PyTorch, etc.)
- [x] Setup device configuration (CUDA/CPU)

### DINOv2 Setup
- [x] Clone DINOv2 repository
- [x] Add to Python path
- [x] Load ViT-B model via torch.hub
- [x] Implement feature extraction utility
- [x] Test inference with dummy image

### DINOv3 Setup
- [x] Clone DINOv3 repository
- [x] Add to Python path
- [x] Configure checkpoint directory
- [ ] **TODO**: Request checkpoint access
- [ ] **TODO**: Download ViT-B checkpoint
- [ ] **TODO**: Implement checkpoint loader
- [ ] **TODO**: Test inference

### SAM Setup
- [x] Install segment-anything package
- [x] Configure checkpoint directory
- [x] Download ViT-B checkpoint
- [x] Load SAM model
- [x] Implement feature extraction utility
- [x] Test inference with dummy image

### Dataset Setup
- [x] Clone SD4Match repository
- [x] Add to Python path
- [x] Configure data directory
- [ ] **TODO**: Download SD4Match dataset
- [ ] **TODO**: Verify data splits (trn/val/test)
- [ ] **TODO**: Test dataset loading

### Utilities & Documentation
- [x] Create configuration class
- [x] Implement visualization utilities
- [x] Implement checkpoint save/load functions
- [x] Create setup guide (SETUP_GUIDE.md)
- [x] Create quick reference (QUICK_REFERENCE.md)
- [x] Document professor's requirements

---

## Phase 2: Dataset Integration (Team TODO)

### SD4Match Integration
- [ ] Implement dataset loader
- [ ] Create data preprocessing pipeline
- [ ] Setup data augmentation
- [ ] Create train/val/test dataloaders
- [ ] Verify image loading and transforms

### Evaluation Metrics
- [ ] Implement PCK (Percentage of Correct Keypoints)
- [ ] Implement other SD4Match metrics
- [ ] Create evaluation pipeline
- [ ] Test metrics on sample data

---

## Phase 3: Feature Extraction Pipeline (Team TODO)

### DINOv2 Features
- [ ] Batch processing implementation
- [ ] Memory-efficient feature extraction
- [ ] Feature normalization
- [ ] Cache features to disk (optional)

### DINOv3 Features
- [ ] Complete model loading
- [ ] Batch processing implementation
- [ ] Feature extraction pipeline
- [ ] Compare with DINOv2 outputs

### SAM Features
- [ ] Batch processing implementation
- [ ] Handle different image sizes
- [ ] Feature post-processing
- [ ] Integration with other backbones

---

## Phase 4: Correspondence Methods (Team TODO)

### Basic Matching
- [ ] Implement nearest neighbor matching
- [ ] Implement mutual nearest neighbors
- [ ] Feature similarity metrics
- [ ] Initial correspondence estimation

### Advanced Methods
- [ ] Study GeoAware-SC paper
- [ ] Clone GeoAware-SC repository
- [ ] Integrate Window Soft Argmax
- [ ] Implement refinement pipeline

### Multi-backbone Fusion
- [ ] Combine features from multiple backbones
- [ ] Ensemble predictions
- [ ] Weighted voting strategies

---

## Phase 5: Training & Validation (Team TODO)

### Training Setup
- [ ] Define loss functions
- [ ] Setup optimizer and scheduler
- [ ] Implement training loop
- [ ] Add logging (tensorboard/wandb)
- [ ] Implement early stopping

### Validation
- [ ] Validation loop implementation
- [ ] Model selection criteria
- [ ] Hyperparameter tuning
- [ ] Save best models

### Experiments
- [ ] Baseline experiments (single backbone)
- [ ] Compare DINOv2 vs DINOv3 vs SAM
- [ ] Test backbone size variations
- [ ] Ablation studies

---

## Phase 6: Final Evaluation (Team TODO)

### Test Evaluation
- [ ] ⚠️ **IMPORTANT**: Evaluate ONLY on test split
- [ ] Run final model on test set
- [ ] Compute all metrics
- [ ] Generate visualizations
- [ ] Error analysis

### Results Documentation
- [ ] Create results tables
- [ ] Generate comparison plots
- [ ] Analyze backbone performance
- [ ] Document findings

---

## Phase 7: Reporting & Submission (Team TODO)

### Code Organization
- [ ] Clean up code
- [ ] Add comprehensive comments
- [ ] Write documentation
- [ ] Create README for submission

### Report
- [ ] Write methodology section
- [ ] Document experiments
- [ ] Present results with tables/figures
- [ ] Discuss findings
- [ ] Conclusion and future work

### Submission
- [ ] Review submission requirements
- [ ] Package code and models
- [ ] Final testing
- [ ] Submit project

---

## Optional Enhancements (If Time Permits)

### Model Comparisons
- [ ] Test ViT-S (Small) backbones
- [ ] Test ViT-L (Large) backbones
- [ ] Performance vs. size analysis
- [ ] Speed benchmarking

### Advanced Features
- [ ] Data augmentation strategies
- [ ] Self-supervised learning
- [ ] Transfer learning experiments
- [ ] Cross-dataset evaluation

### Visualization
- [ ] Interactive correspondence viewer
- [ ] Attention map visualization
- [ ] Error case analysis tools
- [ ] Demo notebook

---

## Meeting Notes & Decisions

### Professor's Requirements
- ✅ Use Base (ViT-B) models as primary
- ✅ Official repos, not just Hugging Face
- ✅ Strict train/val/test split usage
- ✅ Final results ONLY on test split
- ⚠️ Size comparison optional but recommended

### Team Decisions
- [ ] **TODO**: Assign phase responsibilities
- [ ] **TODO**: Set milestone deadlines
- [ ] **TODO**: Choose communication tools
- [ ] **TODO**: Define code review process

---

## Immediate Action Items

### Priority 1 (Critical)
- [ ] **Request DINOv3 checkpoint access** (whoever)
- [ ] **Download SD4Match dataset** (whoever)

### Priority 2 (High)
- [ ] Complete DINOv3 integration
- [ ] Test all models with real images
- [ ] Implement dataset loader

### Priority 3 (Medium)
- [ ] Start correspondence implementation
- [ ] Setup experiment tracking
- [ ] Begin baseline experiments

---

## Resources & Links

### Documentation
- [x] SETUP_GUIDE.md
- [x] QUICK_REFERENCE.md
- [x] This checklist (PROJECT_CHECKLIST.md)

### External Resources
- SD4Match: https://github.com/ActiveVisionLab/SD4Match
- DINOv2: https://github.com/facebookresearch/dinov2
- DINOv3: https://github.com/facebookresearch/dinov3
- SAM: https://github.com/facebookresearch/segment-anything
- GeoAware-SC: https://github.com/Junyi42/geoaware-sc

---

**Last Updated**: December 10, 2025  
**Phase Status**: Phase 1 Complete ✅  
**Next Milestone**: Complete manual setup steps and begin Phase 2
