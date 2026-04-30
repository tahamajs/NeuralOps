# studio/visualization/dashboard.py
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from transformers import AutoTokenizer
import os

class AttentionDashboard:
    """
    Interactive dashboard to visualise attention, hidden states, gradients, and embeddings.
    """
    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self._hooks = []
        self._attention_maps = []
        self._hidden_states = []
        self._gradients = []

    def _register_hooks(self, capture_attn=True, capture_hidden=True, capture_grad=False):
        def attn_hook(module, inp, out, name):
            if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
                self._attention_maps.append((name, out[1].detach().cpu()))
        def hidden_hook(module, inp, out, name):
            self._hidden_states.append((name, out.detach().cpu()))
        def grad_hook(module, grad_in, grad_out, name):
            self._gradients.append((name, grad_out[0].detach().cpu() if grad_out else None))

        for n, m in self.model.named_modules():
            if capture_attn and ('attention' in n.lower() or 'attn' in n.lower()):
                self._hooks.append(m.register_forward_hook(lambda m,i,o, n=n: attn_hook(m,i,o,n)))
            if capture_hidden and ('layer' in n.lower() or 'encoder' in n.lower()):
                self._hooks.append(m.register_forward_hook(lambda m,i,o, n=n: hidden_hook(m,i,o,n)))
            if capture_grad and isinstance(m, torch.nn.Linear):
                self._hooks.append(m.register_full_backward_hook(lambda m,gi,go, n=n: grad_hook(m,gi,go,n)))

    def _clear_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._attention_maps = []
        self._hidden_states = []
        self._gradients = []

    def analyze_text(self, text, layer=0, head=0, output_html="attention_dashboard.html"):
        """
        Run forward+backward on a text and generate interactive visualisations.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        self._register_hooks(capture_attn=True, capture_hidden=True, capture_grad=True)
        self.model.zero_grad()
        outputs = self.model(**inputs)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs.logits.sum()
        loss.backward()
        self._clear_hooks()

        # Prepare plots
        fig = make_subplots(rows=3, cols=2,
                            subplot_titles=("Attention Heatmap", "Attention Entropy",
                                            "Hidden State Similarity", "Gradient Norms",
                                            "Embedding Projection (PCA)", "Layer Importance"),
                            specs=[[{'type': 'heatmap'}, {'type': 'scatter'}],
                                   [{'type': 'scatter'}, {'type': 'bar'}],
                                   [{'type': 'scatter'}, {'type': 'bar'}]])
        # 1. Attention heatmap for a specific layer/head
        attn_name, attn_map = self._attention_maps[layer] if layer < len(self._attention_maps) else (None, None)
        if attn_map is not None:
            attn_matrix = attn_map[0, head].numpy()  # (seq, seq)
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            fig.add_trace(go.Heatmap(z=attn_matrix, x=tokens, y=tokens, colorscale='Viridis'), row=1, col=1)

        # 2. Attention entropy per token
        if attn_map is not None:
            entropy = -np.sum(attn_matrix * np.log(attn_matrix + 1e-10), axis=-1)
            fig.add_trace(go.Scatter(x=list(range(len(entropy))), y=entropy, mode='lines+markers'), row=1, col=2)

        # 3. Cosine similarity between consecutive hidden states
        if len(self._hidden_states) > 1:
            sims = []
            for i in range(len(self._hidden_states)-1):
                h1 = self._hidden_states[i][1].mean(dim=(0,1)).numpy()
                h2 = self._hidden_states[i+1][1].mean(dim=(0,1)).numpy()
                sim = np.dot(h1, h2) / (np.linalg.norm(h1)*np.linalg.norm(h2)+1e-8)
                sims.append(sim)
            fig.add_trace(go.Scatter(x=list(range(len(sims))), y=sims, mode='lines+markers'), row=2, col=1)

        # 4. Gradient norms per layer (absolute mean)
        if self._gradients:
            layer_names = [n for n,_ in self._gradients]
            grad_norms = [g.abs().mean().item() if g is not None else 0 for _,g in self._gradients]
            fig.add_trace(go.Bar(x=layer_names, y=grad_norms), row=2, col=2)

        # 5. PCA of hidden states (first layer vs last layer)
        if len(self._hidden_states) >= 2:
            from sklearn.decomposition import PCA
            first_hid = self._hidden_states[0][1].view(-1, self._hidden_states[0][1].shape[-1]).numpy()
            last_hid = self._hidden_states[-1][1].view(-1, self._hidden_states[-1][1].shape[-1]).numpy()
            pca = PCA(n_components=2)
            first_2d = pca.fit_transform(first_hid)
            last_2d = pca.transform(last_hid)
            fig.add_trace(go.Scatter(x=first_2d[:,0], y=first_2d[:,1], mode='markers', name='First layer'), row=3, col=1)
            fig.add_trace(go.Scatter(x=last_2d[:,0], y=last_2d[:,1], mode='markers', name='Last layer'), row=3, col=1)

        # 6. Layer importance (average attention received by CLS token)
        if attn_map is not None:
            cls_attn = attn_matrix[0, :]  # first token is CLS
            fig.add_trace(go.Bar(x=tokens, y=cls_attn, name='Attention to CLS'), row=3, col=2)

        fig.update_layout(height=1200, width=1500, title_text=f"Model Analysis for: {text[:50]}...")
        fig.write_html(output_html)
        print(f"Dashboard saved to {output_html}")
        return fig

class ModelVisualizer:
    @staticmethod
    def plot_training_curves(log_history, save_path="training_curves.html"):
        import pandas as pd
        df = pd.DataFrame(log_history)
        fig = make_subplots(rows=1, cols=2)
        if 'loss' in df.columns:
            fig.add_trace(go.Scatter(x=df['step'], y=df['loss'], name='Train Loss'), row=1, col=1)
        if 'eval_loss' in df.columns:
            eval_steps = df[df['eval_loss'].notna()]
            fig.add_trace(go.Scatter(x=eval_steps['step'], y=eval_steps['eval_loss'], name='Val Loss'), row=1, col=1)
        if 'eval_accuracy' in df.columns:
            eval_acc = df[df['eval_accuracy'].notna()]
            fig.add_trace(go.Scatter(x=eval_acc['step'], y=eval_acc['eval_accuracy'], name='Accuracy'), row=1, col=2)
        fig.write_html(save_path)
        print(f"Training curves saved to {save_path}")