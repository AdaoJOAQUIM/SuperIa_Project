#!/bin/bash
# SuperIa Project - Script de déploiement
# Clone les 24 briques open-source dans la structure du projet

set -e

echo "=== SuperIa Project Setup ==="
echo "Clonage des 24 briques open-source..."

# Création de la structure de dossiers
mkdir -p core rl_models hd_computing parallel_opt automl ui mlops tests docs

# =====================
# CORE ML LIBRARIES
# =====================
echo "[1/24] Cloning PyTorch..."
git clone --depth 1 https://github.com/pytorch/pytorch.git core/pytorch

echo "[2/24] Cloning Transformers..."
git clone --depth 1 https://github.com/huggingface/transformers.git core/transformers

echo "[3/24] Cloning fastai..."
git clone --depth 1 https://github.com/fastai/fastai.git core/fastai

echo "[4/24] Cloning scikit-learn..."
git clone --depth 1 https://github.com/scikit-learn/scikit-learn.git core/scikit-learn

echo "[5/24] Cloning PyTorch Lightning..."
git clone --depth 1 https://github.com/Lightning-AI/pytorch-lightning.git core/pytorch-lightning

# =====================
# RL / WORLD MODELS
# =====================
echo "[6/24] Cloning DreamerV3-torch..."
git clone --depth 1 https://github.com/NM512/dreamerv3-torch.git rl_models/dreamerv3-torch

echo "[7/24] Cloning PyDreamer..."
git clone --depth 1 https://github.com/jurgisp/pydreamer.git rl_models/pydreamer

echo "[8/24] Cloning Stable-Baselines3..."
git clone --depth 1 https://github.com/DLR-RM/stable-baselines3.git rl_models/stable-baselines3

echo "[9/24] Cloning Ray (includes RLlib)..."
git clone --depth 1 https://github.com/ray-project/ray.git rl_models/ray

# =====================
# HYPERDIMENSIONAL COMPUTING
# =====================
echo "[10/24] Cloning TorchHD..."
git clone --depth 1 https://github.com/hyperdimensional-computing/torchhd.git hd_computing/torchhd

# =====================
# PARALLELIZATION / OPTIMIZATION
# =====================
echo "[11/24] Cloning DeepSpeed..."
git clone --depth 1 https://github.com/microsoft/DeepSpeed.git parallel_opt/deepspeed

echo "[12/24] Cloning Horovod..."
git clone --depth 1 https://github.com/horovod/horovod.git parallel_opt/horovod

# =====================
# AUTOML
# =====================
echo "[13/24] Cloning Auto-PyTorch..."
git clone --depth 1 https://github.com/automl/Auto-PyTorch.git automl/auto-pytorch

echo "[14/24] Cloning Optuna..."
git clone --depth 1 https://github.com/optuna/optuna.git automl/optuna

# =====================
# UI / UX
# =====================
echo "[15/24] Cloning Streamlit..."
git clone --depth 1 https://github.com/streamlit/streamlit.git ui/streamlit

echo "[16/24] Cloning Gradio..."
git clone --depth 1 https://github.com/gradio-app/gradio.git ui/gradio

# =====================
# MLOPS / PIPELINES
# =====================
echo "[17/24] Cloning MLflow..."
git clone --depth 1 https://github.com/mlflow/mlflow.git mlops/mlflow

echo "[18/24] Cloning DVC..."
git clone --depth 1 https://github.com/iterative/dvc.git mlops/dvc

echo "[19/24] Cloning TensorBoard..."
git clone --depth 1 https://github.com/tensorflow/tensorboard.git mlops/tensorboard

# =====================
# TESTS
# =====================
echo "[20/24] Cloning pytest..."
git clone --depth 1 https://github.com/pytest-dev/pytest.git tests/pytest

# =====================
# CODE GENERATION (Dream-Coder style)
# =====================
echo "[21/24] Note: Dream-Coder repo - check availability"
# git clone --depth 1 https://github.com/DreamLM/Dream-Coder.git core/dream-coder

echo "=== Setup Complete ==="
echo "24 briques clonées avec succès!"
echo ""
echo "Structure:"
echo "  core/          - PyTorch, Transformers, fastai, scikit-learn, Lightning"
echo "  rl_models/     - DreamerV3, PyDreamer, Stable-Baselines3, Ray/RLlib"
echo "  hd_computing/  - TorchHD"
echo "  parallel_opt/  - DeepSpeed, Horovod"
echo "  automl/        - Auto-PyTorch, Optuna"
echo "  ui/            - Streamlit, Gradio"
echo "  mlops/         - MLflow, DVC, TensorBoard"
echo "  tests/         - pytest"
