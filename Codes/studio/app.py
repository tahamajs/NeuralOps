#!/usr/bin/env python
# studio/app.py
"""
SVD-Lin Studio - Complete Transformer Research Platform
Interactive web interface for exploring and modifying transformer models
"""

import streamlit as st
import torch
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import os

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from studio.core.model_manager import ModelManager
from studio.core.attention_manager import AttentionManager
from studio.core.compression_manager import CompressionManager
from studio.core.visualization_manager import VisualizationManager
from studio.ui.pages.model_explorer import ModelExplorer
from studio.ui.pages.attention_editor import AttentionEditor
from studio.ui.pages.compression_lab import CompressionLab
from studio.ui.pages.visualization_studio import VisualizationStudio
from studio.ui.pages.training_pipeline import TrainingPipeline
from studio.ui.pages.experiment_tracker import ExperimentTracker


# Page configuration
st.set_page_config(
    page_title="SVD-Lin Studio - Transformer Research Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem;
        text-align: center;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f0f2f6 0%, #e8eaef 100%);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = ModelManager()
    if 'attention_manager' not in st.session_state:
        st.session_state.attention_manager = AttentionManager()
    if 'compression_manager' not in st.session_state:
        st.session_state.compression_manager = CompressionManager()
    if 'visualization_manager' not in st.session_state:
        st.session_state.visualization_manager = VisualizationManager()
    if 'current_model' not in st.session_state:
        st.session_state.current_model = None
    if 'current_tokenizer' not in st.session_state:
        st.session_state.current_tokenizer = None
    if 'experiments' not in st.session_state:
        st.session_state.experiments = []


def sidebar():
    """Render sidebar navigation."""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
        st.markdown("## 🔬 SVD-Lin Studio")
        st.markdown("*Transformer Research Platform*")
        st.markdown("---")
        
        # Navigation
        pages = {
            "🏠 Model Explorer": "model_explorer",
            "🎯 Attention Editor": "attention_editor",
            "🗜️ Compression Lab": "compression_lab",
            "📊 Visualization Studio": "visualization_studio",
            "🏋️ Training Pipeline": "training_pipeline",
            "📈 Experiment Tracker": "experiment_tracker",
        }
        
        selection = st.radio("Navigation", list(pages.keys()))
        st.session_state.current_page = pages[selection]
        
        st.markdown("---")
        
        # Model info if loaded
        if st.session_state.current_model:
            st.markdown("### Current Model")
            st.info(f"**Model:** {st.session_state.current_model.config._name_or_path if hasattr(st.session_state.current_model, 'config') else 'Custom'}")
            
            total_params = sum(p.numel() for p in st.session_state.current_model.parameters())
            st.metric("Parameters", f"{total_params:,}")
        
        st.markdown("---")
        st.markdown("### Quick Links")
        st.markdown("- [Documentation](https://github.com/your-repo/svd-lin)")
        st.markdown("- [Paper](https://arxiv.org/abs/...)")


def main():
    """Main application entry point."""
    st.markdown('<div class="main-header">🔬 SVD-Lin Studio</div>', unsafe_allow_html=True)
    st.markdown("*Complete Platform for Transformer Model Research and Optimization*")
    st.markdown("---")
    
    init_session_state()
    sidebar()
    
    # Route to selected page
    page = st.session_state.get('current_page', 'model_explorer')
    
    if page == "model_explorer":
        ModelExplorer.render()
    elif page == "attention_editor":
        AttentionEditor.render()
    elif page == "compression_lab":
        CompressionLab.render()
    elif page == "visualization_studio":
        VisualizationStudio.render()
    elif page == "training_pipeline":
        TrainingPipeline.render()
    elif page == "experiment_tracker":
        ExperimentTracker.render()


if __name__ == "__main__":
    main()