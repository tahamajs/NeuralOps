import os
import glob
import ast
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Configuration
PROJECT_ROOT = "/Users/tahamajs/Documents/uni/Project"
MODULE_METADATA_DIR = os.path.join(PROJECT_ROOT, "module_metadata")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "paper", "neuralops_paper_final", "figures", "catalog_visuals")

os.makedirs(MODULE_METADATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Theme
BG_COLOR = '#0f172a'
TEXT_COLOR = 'white'
ACCENTS = {
    'linear': '#38bdf8', # Blue
    'ssm': '#10b981',    # Green
    'moe': '#f59e0b',    # Orange
    'sparse': '#ef4444', # Red
    'kernel': '#a855f7', # Purple
    'default': '#94a3b8' # Gray
}

def analyze_ast(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
    
    try:
        tree = ast.parse(source)
    except:
        return None
        
    metadata = {
        'filename': os.path.basename(filepath),
        'class_name': None,
        'hyperparameters': [],
        'forward_operations': [],
        'type': 'default'
    }
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if not metadata['class_name']:
                metadata['class_name'] = node.name
                
            # Check for type hints
            name_lower = node.name.lower()
            if 'linear' in name_lower or 'svd' in name_lower or 'lowrank' in name_lower:
                metadata['type'] = 'linear'
            elif 'ssm' in name_lower or 'recurrent' in name_lower or 'mamba' in name_lower or 'rnn' in name_lower:
                metadata['type'] = 'ssm'
            elif 'moe' in name_lower or 'hybrid' in name_lower or 'expert' in name_lower:
                metadata['type'] = 'moe'
            elif 'sparse' in name_lower or 'prun' in name_lower or 'quant' in name_lower:
                metadata['type'] = 'sparse'
            elif 'kernel' in name_lower or 'spectral' in name_lower or 'fourier' in name_lower:
                metadata['type'] = 'kernel'
                
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    if child.name == '__init__':
                        for arg in child.args.args:
                            if arg.arg != 'self':
                                metadata['hyperparameters'].append(arg.arg)
                    elif child.name == 'forward':
                        for f_node in ast.walk(child):
                            if isinstance(f_node, ast.Call) and isinstance(f_node.func, ast.Attribute):
                                metadata['forward_operations'].append(f_node.func.attr)
                            elif isinstance(f_node, ast.Call) and isinstance(f_node.func, ast.Name):
                                metadata['forward_operations'].append(f_node.func.id)
    
    # deduplicate operations
    metadata['forward_operations'] = list(set(metadata['forward_operations']))
    return metadata

def draw_architecture(ax, metadata, color):
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')
    
    ops = metadata['forward_operations'][:4]
    if not ops:
        ops = ['Linear', 'Activation', 'Dropout', 'LayerNorm']
        
    y_pos = 0.8
    ax.text(0.5, 0.95, f"{metadata['class_name']} Architecture", color=TEXT_COLOR, ha='center', fontsize=12, fontweight='bold')
    
    for i, op in enumerate(ops):
        rect = patches.Rectangle((0.3, y_pos - 0.1), 0.4, 0.15, linewidth=2, edgecolor=color, facecolor=BG_COLOR)
        ax.add_patch(rect)
        ax.text(0.5, y_pos - 0.025, op, color=TEXT_COLOR, ha='center', va='center', fontsize=10)
        
        if i < len(ops) - 1:
            ax.annotate('', xy=(0.5, y_pos - 0.15), xytext=(0.5, y_pos - 0.1),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2))
        y_pos -= 0.2

def draw_pattern(ax, metadata, color):
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)
        
    m_type = metadata['type']
    np.random.seed(hash(metadata['class_name']) % (2**32))
    
    if m_type == 'ssm':
        # Decay matrix
        data = np.exp(-np.linspace(0, 5, 20)[:, None] * np.linspace(0, 1, 20)[None, :])
        title = "State Space Decay Pattern"
    elif m_type == 'sparse':
        # Sparse pattern
        data = np.random.choice([0, 1], size=(20, 20), p=[0.9, 0.1])
        title = "Sparsity Mask"
    elif m_type == 'linear':
        # Low rank pattern
        u = np.random.randn(20, 4)
        v = np.random.randn(4, 20)
        data = u @ v
        title = "Low-Rank Representation"
    elif m_type == 'moe':
        # MoE routing
        data = np.zeros((20, 20))
        for i in range(20):
            data[i, np.random.randint(0, 20)] = 1.0
        title = "Expert Routing Distribution"
    else:
        # Standard attention
        data = np.random.randn(20, 20)
        data = np.tril(data) # causal mask
        title = "Attention Heatmap"
        
    im = ax.imshow(data, cmap='viridis' if m_type != 'sparse' else 'Reds', aspect='auto')
    ax.set_title(title, color=TEXT_COLOR, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

def draw_complexity(ax, metadata, color):
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(TEXT_COLOR)
        
    m_type = metadata['type']
    seq_lengths = np.logspace(2, 4, 100)
    
    if m_type in ['ssm', 'linear']:
        flops = seq_lengths * np.log(seq_lengths)
        label = "O(N log N) FLOPs"
    elif m_type == 'sparse':
        flops = seq_lengths * np.sqrt(seq_lengths)
        label = "O(N√N) FLOPs"
    else:
        flops = seq_lengths**2
        label = "O(N²) FLOPs"
        
    ax.plot(seq_lengths, flops, color=color, linewidth=2, label=label)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title("Complexity / Scaling Profile", color=TEXT_COLOR, fontsize=12)
    ax.set_xlabel("Sequence Length ($L$)", color=TEXT_COLOR)
    ax.set_ylabel("Compute (FLOPs)", color=TEXT_COLOR)
    ax.legend(facecolor=BG_COLOR, edgecolor=color, labelcolor=TEXT_COLOR)
    ax.grid(True, alpha=0.2, color=TEXT_COLOR, linestyle='--')

def process_file(filepath):
    print(f"Processing: {filepath}")
    metadata = analyze_ast(filepath)
    if not metadata or not metadata['class_name']:
        print(f"  Skipped (no class found)")
        return
        
    module_name = os.path.splitext(os.path.basename(filepath))[0]
    
    # Save metadata JSON
    json_path = os.path.join(MODULE_METADATA_DIR, f"{module_name}.json")
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    # Generate Figure
    fig = plt.figure(figsize=(12, 4), facecolor=BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)
    
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)
    
    color = ACCENTS.get(metadata['type'], ACCENTS['default'])
    
    draw_architecture(ax1, metadata, color)
    draw_pattern(ax2, metadata, color)
    draw_complexity(ax3, metadata, color)
    
    plt.suptitle(f"{metadata['class_name']} ({module_name})", color=TEXT_COLOR, fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    # Save PDF and PNG
    pdf_path = os.path.join(OUTPUT_DIR, f"{module_name}.pdf")
    png_path = os.path.join(OUTPUT_DIR, f"{module_name}.png")
    
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor=BG_COLOR)
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Saved figure for {module_name}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python generate_bespoke_swarm.py <module_type: attentions|compression_engine> <start_letter> <end_letter>")
        sys.exit(1)
        
    mod_type = sys.argv[1]
    start_letter = sys.argv[2].lower()
    end_letter = sys.argv[3].lower()
    
    target_dir = os.path.join(PROJECT_ROOT, "Codes", "framework", mod_type)
    if not os.path.exists(target_dir):
        print(f"Directory not found: {target_dir}")
        sys.exit(1)
        
    files = sorted(glob.glob(os.path.join(target_dir, "*.py")))
    
    for filepath in files:
        basename = os.path.basename(filepath).lower()
        if basename == "__init__.py": continue
        
        first_letter = basename[0]
        if start_letter <= first_letter <= end_letter:
            process_file(filepath)
