#!/usr/bin/env bash
#SBATCH --job-name=isaaclab-cartpole
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%j.out

set -euo pipefail
# The gpu partition/resource syntax is site-specific. See hpc/README.md.
: "${ISAACLAB_ROOT:?Set ISAACLAB_ROOT to a compatible local IsaacLab 3.0 checkout}"
cd "$ISAACLAB_ROOT"
export UV_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/uv-cache-${USER}"
mkdir -p "$UV_CACHE_DIR" logs
nvidia-smi
uv run isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct \
  --num_envs 1024 --max_iterations 200 physics=newton_mjwarp --viz none
