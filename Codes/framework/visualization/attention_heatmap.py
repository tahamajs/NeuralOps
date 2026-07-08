# framework/visualization/attention_heatmap.py
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from .base import VISUALIZER_REGISTRY, BaseVisualizer

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


@VISUALIZER_REGISTRY.register("attention_heatmap")
class AttentionHeatmapVisualizer(BaseVisualizer):
    def __init__(self, config=None, output_dir="visualizations"):
        super().__init__(output_dir)
        self.config = config or {}
        self.defaults = {
            "cmap": "viridis",
            "normalize": "none",
            "aggregate_heads": "mean",
            "output_format": "matplotlib",
            "figsize": (10, 8),
            "dpi": 150,
            "show_values": False,
            "fmt": ".2f",
            "annot_fontsize": 8,
            "tick_fontsize": 10,
            "title": None,
            "xlabel": "Key Position",
            "ylabel": "Query Position",
            "rotate_xticks": 45,
            "rotate_yticks": 0,
            "colorbar": True,
            "square": True,
            "mask_diagonal": False,
            "crop": None,
            "head_layout": "auto",
            "save_ext": "png",
        }
        self.defaults.update(self.config)

    def generate(self, attention_matrix, output_path=None, **kwargs):
        opts = self.defaults.copy()
        opts.update(kwargs)

        if torch.is_tensor(attention_matrix):
            attention_matrix = attention_matrix.detach().cpu().numpy()
        att = attention_matrix

        # Input shape handling
        if att.ndim == 3:
            n_heads, seq_len, _ = att.shape
            if opts["aggregate_heads"]:
                att = getattr(np, opts["aggregate_heads"])(att, axis=0)
                is_multihead = False
            else:
                is_multihead = True
        else:
            is_multihead = False
            n_heads = 1

        # Crop
        if opts["crop"] is not None:
            sl = (
                slice(0, opts["crop"])
                if isinstance(opts["crop"], int)
                else slice(opts["crop"][0], opts["crop"][1])
            )
            att = att[sl, sl] if not is_multihead else att[:, sl, sl]

        # Normalize
        att = self._normalize(att, opts["normalize"])
        if opts["mask_diagonal"] and not is_multihead:
            np.fill_diagonal(att, np.nan)

        # Output
        if opts["output_format"] == "plotly" and PLOTLY_AVAILABLE:
            return self._plotly_heatmap(att, output_path, opts, is_multihead)
        else:
            return self._matplotlib_heatmap(
                att, output_path, opts, is_multihead, n_heads
            )

    def _normalize(self, att, method):
        if method == "none":
            return att
        if method == "row":
            row_sums = att.sum(axis=-1, keepdims=True)
            return att / np.where(row_sums == 0, 1, row_sums)
        if method == "col":
            col_sums = att.sum(axis=-2, keepdims=True)
            return att / np.where(col_sums == 0, 1, col_sums)
        if method == "softmax":
            exp_att = np.exp(att - att.max())
            return exp_att / exp_att.sum()
        raise ValueError(f"Unknown normalization: {method}")

    def _matplotlib_heatmap(self, att, output_path, opts, is_multihead, n_heads):
        sns.set_theme(style="white", font_scale=1.0)
        if not is_multihead:
            fig, ax = plt.subplots(figsize=opts["figsize"], dpi=opts["dpi"])
            self._draw(ax, att, opts)
            ax.set_title(opts["title"] or "")
            ax.set_xlabel(opts["xlabel"])
            ax.set_ylabel(opts["ylabel"])
            self._apply_tick_labels(ax, att.shape[0], opts)
            return self._handle_output(fig, output_path, opts)
        else:
            # multi-head grid
            if opts["head_layout"] == "auto":
                cols = max(1, int(math.sqrt(n_heads)))
                rows = (n_heads + cols - 1) // cols
            elif opts["head_layout"] == "horizontal":
                rows, cols = 1, n_heads
            elif opts["head_layout"] == "vertical":
                rows, cols = n_heads, 1
            else:
                rows = cols = int(math.ceil(math.sqrt(n_heads)))
            fig, axes = plt.subplots(
                rows,
                cols,
                figsize=(opts["figsize"][0] * cols, opts["figsize"][1] * rows),
                dpi=opts["dpi"],
                squeeze=False,
            )
            for i in range(n_heads):
                r, c = i // cols, i % cols
                self._draw(axes[r, c], att[i], opts)
                axes[r, c].set_title(f"Head {i}", fontsize=opts["tick_fontsize"])
                axes[r, c].set_xlabel("Key")
                axes[r, c].set_ylabel("Query")
            for i in range(n_heads, rows * cols):
                axes[i // cols, i % cols].axis("off")
            fig.suptitle(
                opts["title"] or "Attention Heads", fontsize=opts["tick_fontsize"] + 2
            )
            plt.tight_layout()
            if output_path:
                base, ext = os.path.splitext(output_path)
                out_path = base + "_heads" + (ext or ".png")
            else:
                out_path = None
            return self._handle_output(fig, out_path, opts)

    def _draw(self, ax, data, opts):
        mask = np.eye(data.shape[0], dtype=bool) if opts["mask_diagonal"] else None
        sns.heatmap(
            data,
            ax=ax,
            cmap=opts["cmap"],
            annot=opts["show_values"],
            fmt=opts["fmt"],
            annot_kws={"size": opts["annot_fontsize"]},
            cbar=opts["colorbar"],
            square=opts["square"],
            mask=mask,
            xticklabels=False,
            yticklabels=False,
        )

    def _apply_tick_labels(self, ax, n, opts):
        if n <= 20 and not opts.get("hide_ticks", False):
            ax.set_xticks(np.arange(n) + 0.5)
            ax.set_xticklabels(
                range(n), rotation=opts["rotate_xticks"], fontsize=opts["tick_fontsize"]
            )
            ax.set_yticks(np.arange(n) + 0.5)
            ax.set_yticklabels(
                range(n), rotation=opts["rotate_yticks"], fontsize=opts["tick_fontsize"]
            )
        else:
            ax.set_xticks([])
            ax.set_yticks([])

    def _plotly_heatmap(self, att, output_path, opts, is_multihead):
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly not installed")
        if is_multihead:
            fig = go.Figure()
            for i in range(att.shape[0]):
                fig.add_trace(
                    go.Heatmap(
                        z=att[i],
                        colorscale=opts["cmap"],
                        showscale=opts["colorbar"],
                        name=f"Head {i}",
                        visible=(i == 0),
                    )
                )
            buttons = [
                {
                    "label": f"Head {i}",
                    "method": "update",
                    "args": [
                        {"visible": [j == i for j in range(att.shape[0])]},
                        {"title": f"Attention Head {i}"},
                    ],
                }
                for i in range(att.shape[0])
            ]
            fig.update_layout(
                updatemenus=[{"active": 0, "buttons": buttons}],
                title=opts["title"] or "Attention Heads",
                xaxis_title=opts["xlabel"],
                yaxis_title=opts["ylabel"],
            )
        else:
            fig = go.Figure(
                data=go.Heatmap(
                    z=att, colorscale=opts["cmap"], showscale=opts["colorbar"]
                )
            )
            fig.update_layout(
                title=opts["title"],
                xaxis_title=opts["xlabel"],
                yaxis_title=opts["ylabel"],
            )
        if output_path:
            if opts["save_ext"].lower() == "html":
                fig.write_html(output_path)
            else:
                fig.write_image(output_path, scale=opts["dpi"] / 72)
            return output_path
        return fig

    def _handle_output(self, fig, output_path, opts):
        if output_path:
            path = self._add_extension(output_path, opts["save_ext"])
            fig.savefig(path, bbox_inches="tight", dpi=opts["dpi"])
            plt.close(fig)
            return path
        return fig

    def _add_extension(self, path, ext):
        if not path:
            return None
        if not os.path.splitext(path)[1]:
            path = path + "." + ext.lstrip(".")
        return path
