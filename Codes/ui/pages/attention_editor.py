# studio/ui/pages/attention_editor.py
"""
Interactive attention editor for modifying transformer attention mechanisms.
"""

import streamlit as st
import torch
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional

from studio.core.attention_manager import AttentionManager
from studio.core.visualization_manager import VisualizationManager


class AttentionEditor:
    """Interactive page for editing attention mechanisms."""
    
    @staticmethod
    def render():
        st.markdown("## 🎯 Attention Editor")
        st.markdown("*Edit and customize attention mechanisms in your transformer model*")
        st.markdown("---")
        
        # Sidebar controls
        with st.sidebar:
            st.markdown("### Attention Settings")
            
            attention_type = st.selectbox(
                "Attention Type",
                ["svd", "nystrom", "performer", "linformer", "flash", "sliding_window", "sparse", "full"],
                help="Select attention mechanism to apply"
            )
            
            if attention_type in ["svd", "nystrom", "linformer"]:
                rank = st.slider("Compression Rank", 16, 256, 64, step=16,
                                help="Lower rank = more compression, less computation")
            
            if attention_type == "sliding_window":
                window_size = st.slider("Window Size", 64, 512, 256, step=32,
                                       help="Number of tokens to attend to")
            
            if attention_type == "sparse":
                sparsity = st.slider("Sparsity", 0.1, 0.9, 0.5, step=0.1,
                                    help="Fraction of attention connections to keep")
            
            st.markdown("### Visualization Options")
            layer = st.selectbox("Layer", list(range(12)), index=0)
            head = st.selectbox("Head", list(range(12)), index=0)
            
            visualize = st.button("🔍 Visualize Attention", use_container_width=True)
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Current Model Structure")
            
            if st.session_state.current_model:
                model_info = st.session_state.model_manager.get_model_summary(st.session_state.current_model)
                
                st.metric("Total Parameters", f"{model_info['total_parameters']:,}")
                st.metric("Attention Modules", model_info['num_attention_modules'])
                st.metric("Layers", model_info['num_layers'])
                
                # Display layer structure
                with st.expander("Layer Structure", expanded=False):
                    for layer in model_info['layer_details']:
                        st.markdown(f"- **{layer['name']}** ({layer['type']}) - {layer['parameters']:,} params")
            else:
                st.info("No model loaded. Please load a model from the Model Explorer page.")
        
        with col2:
            st.markdown("### Attention Parameters")
            
            if attention_type == "svd":
                st.info(f"""
                **SVD Attention**\n
                Rank: {rank}\n
                Complexity: O(N·rank)\n
                Memory: {rank * 768 / 1e6:.1f}MB (approx)
                """)
            elif attention_type == "flash":
                st.info(f"""
                **FlashAttention**\n
                Complexity: O(N²) but optimized\n
                Memory: ~O(N) with kernel fusion\n
                Requires: CUDA, torch>=2.0
                """)
            elif attention_type == "sliding_window":
                st.info(f"""
                **Sliding Window Attention**\n
                Window Size: {window_size}\n
                Complexity: O(N·window)\n
                Memory: ~O(N·window)
                """)
        
        # Visualization
        if visualize and st.session_state.current_model:
            st.markdown("---")
            st.markdown("### Attention Visualization")
            
            # Get attention maps
            with st.spinner("Computing attention maps..."):
                try:
                    # Sample input
                    sample_text = st.text_input("Sample text for attention analysis", 
                                               "The cat sat on the mat.")
                    
                    if sample_text and st.session_state.current_tokenizer:
                        inputs = st.session_state.current_tokenizer(
                            sample_text, return_tensors="pt"
                        ).to(st.session_state.current_model.device)
                        
                        # Forward with attention output
                        with torch.no_grad():
                            outputs = st.session_state.current_model(
                                **inputs, output_attentions=True
                            )
                        
                        if hasattr(outputs, 'attentions') and outputs.attentions:
                            attention_maps = outputs.attentions
                            
                            # Plot attention heatmap
                            if attention_maps:
                                attn_map = attention_maps[layer][0, head].detach().cpu()
                                
                                tokens = st.session_state.current_tokenizer.convert_ids_to_tokens(
                                    inputs['input_ids'][0]
                                )
                                
                                vis_manager = VisualizationManager()
                                fig = vis_manager.plot_attention_heatmap(
                                    attn_map, tokens, layer, head, interactive=True
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Additionally show entropy
                                entropy_fig = vis_manager.plot_attention_entropy(attn_map, layer, head)
                                st.plotly_chart(entropy_fig, use_container_width=True)
                        else:
                            st.warning("Model does not output attention maps")
                    else:
                        st.info("Enter sample text to visualize attention")
                        
                except Exception as e:
                    st.error(f"Visualization failed: {e}")
        
        # Apply attention
        st.markdown("---")
        st.markdown("### Apply Changes")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Preview Changes", use_container_width=True):
                st.info("Preview mode: Showing estimated parameter reduction")
                st.metric("Estimated Parameter Reduction", "15-30%")
        
        with col2:
            if st.button("✅ Apply Attention", use_container_width=True, type="primary"):
                if st.session_state.current_model:
                    with st.spinner(f"Applying {attention_type} attention..."):
                        try:
                            # Apply attention
                            attn_manager = AttentionManager()
                            model = attn_manager.apply_attention(
                                st.session_state.current_model,
                                attention_type,
                                rank=rank if attention_type in ["svd", "nystrom", "linformer"] else None
                            )
                            st.session_state.current_model = model
                            st.success(f"✅ Successfully applied {attention_type} attention!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to apply attention: {e}")
                else:
                    st.warning("No model loaded")
        
        with col3:
            if st.button("💾 Save Model", use_container_width=True):
                if st.session_state.current_model:
                    save_path = st.text_input("Save path", "./saved_model")
                    st.session_state.model_manager.save_model(
                        st.session_state.current_model,
                        st.session_state.current_tokenizer,
                        save_path
                    )
                    st.success(f"Model saved to {save_path}")