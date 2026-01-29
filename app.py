"""
SuperIa Project - Streamlit Dashboard
Interface de supervision pour les 24 briques open-source
"""

import streamlit as st

# Configuration de la page
st.set_page_config(
      page_title="SuperIa Dashboard",
      page_icon="🚀",
      layout="wide"
)

# Header
st.title("🚀 SuperIa Project")
st.markdown("Architecture fonctionnelle intégrant 24 briques open-source")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.selectbox(
      "Choisir un module",
      ["Dashboard", "Core ML", "RL Models", "AutoML", "MLOps"]
)

# Main content
if page == "Dashboard":
      st.header("📊 Dashboard Principal")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
              st.metric("Modules Core", "5", "PyTorch, Transformers...")
          with col2:
                    st.metric("Modules RL", "4", "DreamerV3, SB3...")
                with col3:
                          st.metric("AutoML", "2", "Optuna, Auto-PyTorch")
                      with col4:
                                st.metric("MLOps", "3", "MLflow, DVC...")

    st.subheader("🏗️ Architecture")
    st.code("""
    SuperIa_Project/
    ├── core/          # PyTorch, Transformers, fastai
    ├── rl_models/     # DreamerV3, PyDreamer, SB3, RLlib
    ├── hd_computing/  # TorchHD
    ├── parallel_opt/  # Ray, DeepSpeed, Horovod
    ├── automl/        # Auto-PyTorch, Optuna
    ├── ui/            # Streamlit, Gradio
    └── mlops/         # MLflow, DVC, TensorBoard
        """)

elif page == "Core ML":
    st.header("🧠 Core ML Libraries")
    st.write("PyTorch, Transformers, fastai, scikit-learn, Lightning")

elif page == "RL Models":
    st.header("🎮 Reinforcement Learning")
    st.write("DreamerV3, PyDreamer, Stable-Baselines3, RLlib")

elif page == "AutoML":
    st.header("⚡ AutoML")
    st.write("Optuna, Auto-PyTorch")

elif page == "MLOps":
    st.header("🔧 MLOps")
    st.write("MLflow, DVC, TensorBoard")

# Footer
st.markdown("---")
st.markdown("SuperIa Project | MIT License | [GitHub](https://github.com/AdaoJOAQUIM/SuperIa_Project)")
