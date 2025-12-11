Project checkpoints
===================

This repository intentionally does not include large model checkpoints (they exceed GitHub limits). Please download the required checkpoints manually and place them under `checkpoints/` as described below.

Required checkpoints
- `checkpoints/sam/sam_vit_b_01ec64.pth` — SAM ViT-B checkpoint (place under `checkpoints/sam/`)
- DINOv3 checkpoint (if needed) — place under `checkpoints/dinov3/`

How to download
- Obtain checkpoints from the model provider's release page or the project maintainers.
- Example (manual):
  - Create folder: `mkdir -p checkpoints/sam`
  - Move the downloaded file into that folder and rename if necessary.

Using Git LFS (optional)
- If you prefer to keep checkpoints in the repo, use Git LFS. Recommended commands (macOS):

  ```bash
  brew install git-lfs
  git lfs install
  git lfs track "checkpoints/sam/*.pth"
  git add .gitattributes
  git commit -m "Track checkpoints via Git LFS"
  ```

- After tracking, add the checkpoint and push. If the checkpoint was already committed and rejected, follow the migration steps in the README or use the `git lfs migrate` command to rewrite history locally and then force-push (coordinate with collaborators).

Notes for collaborators
- Do not commit checkpoint files directly. Instead, place them under `checkpoints/` locally, or enable Git LFS before cloning.
- If you need help enabling Git LFS on your machine, see: https://git-lfs.github.com/
