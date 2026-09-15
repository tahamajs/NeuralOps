import re

file_path = "/Users/tahamajs/Documents/uni/Project/build_all_folder_plots.py"
with open(file_path, "r", encoding="utf-8") as f:
    code = f.read()

# Make sure plt.close('all') and gc.collect() are called after every figure to prevent Matplotlib RendererAgg leak
addition = """
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    plt.close('all')
    import gc
    gc.collect()
"""

# Replace dpi=300 with dpi=150 and add gc
code = re.sub(r"fig\.savefig\(out_path,\s*dpi=300,\s*bbox_inches='tight'\)\s*plt\.close\(fig\)", addition.strip(), code)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(code)

print("Updated build_all_folder_plots.py with memory safeguards.")
