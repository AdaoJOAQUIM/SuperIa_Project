# SuperIa Project

Architecture fonctionnelle intégrant 24 briques open-source pour RL, HDC, AutoML, MLOps, UI/UX. Déployable 100% web/cloud.

## Architecture

```
SuperIa_Project/
├── core/              # PyTorch, Transformers, fastai, Scikit-learn, Lightning
├── rl_models/         # DreamerV3, PyDreamer, Stable-Baselines3, RLlib
├── hd_computing/      # TorchHD
├── parallel_opt/      # Ray, DeepSpeed, Horovod
├── automl/            # Auto-PyTorch, Optuna
├── ui/                # Streamlit, Gradio
├── mlops/             # MLflow, DVC, TensorBoard
├── tests/             # pytest
├── docs/              # Documentation
└── scripts/           # Pipelines et intégration
```

## Les 24 Briques Open-Source

| # | Fonction | Dépôt |
|---|----------|-------|
| 1 | PyTorch | pytorch/pytorch |
| 2 | Ray | ray-project/ray |
| 3 | DeepSpeed | microsoft/DeepSpeed |
| 4 | DreamerV3 | NM512/dreamerv3-torch |
| 5 | PyDreamer | jurgisp/pydreamer |
| 6 | TorchHD | hyperdimensional-computing/torchhd |
| 7 | Stable-Baselines3 | DLR-RM/stable-baselines3 |
| 8 | Optuna | optuna/optuna |
| 9 | Auto-PyTorch | automl/Auto-PyTorch |
| 10 | Scikit-learn | scikit-learn/scikit-learn |
| 11 | PyTorch Lightning | Lightning-AI/pytorch-lightning |
| 12 | Transformers | huggingface/transformers |
| 13 | Horovod | horovod/horovod |
| 14 | MLflow | mlflow/mlflow |
| 15 | DVC | iterative/dvc |
| 16 | Streamlit | streamlit/streamlit |
| 17 | Gradio | gradio-app/gradio |
| 18 | pytest | pytest-dev/pytest |
| 19 | TensorBoard | tensorflow/tensorboard |
| 20 | fastai | fastai/fastai |

## Déploiement rapide

### Option 1: GitHub Codespaces (recommandé)
1. Cliquez sur "Code" > "Codespaces" > "Create codespace"
2. 2. Exécutez: `chmod +x scripts/setup.sh && ./scripts/setup.sh`
  
   3. ### Option 2: Google Colab
   4. ```python
      !git clone https://github.com/AdaoJOAQUIM/SuperIa_Project.git
      %cd SuperIa_Project
      !chmod +x scripts/setup.sh && ./scripts/setup.sh
      ```

      ### Option 3: Local
      ```bash
      git clone https://github.com/AdaoJOAQUIM/SuperIa_Project.git
      cd SuperIa_Project
      pip install -r requirements.txt
      chmod +x scripts/setup.sh && ./scripts/setup.sh
      ```

      ## Rôles des Composants

      | Composant | Rôle |
      |-----------|------|
      | Claude Chrome | Cloner, organiser, commit & push |
      | Claude Code | Scripts d'intégration & pipelines |
      | Claude IA | Supervision, optimisation, heuristiques |

      ## Déploiement Web/Cloud

      - **Notebooks**: Google Colab (GPU/CPU gratuit)
      - - **UI**: Streamlit Cloud, HuggingFace Spaces
        - - **CI/CD**: GitHub Actions
          - - **MLOps**: MLflow, DVC Studio
           
            - ## License
           
            - MIT License
