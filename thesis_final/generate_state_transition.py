import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

heatmap_path = "img/chart_state_transition_heatmap.png"
if not os.path.exists(heatmap_path):
    # Fake heatmap data resembling Mamba state transition A_t
    np.random.seed(42)
    heatmap_data = np.random.uniform(0.7, 0.99, (32, 32))
    # Add a diagonal structure
    for i in range(32):
        for j in range(32):
            if abs(i-j) < 3:
                heatmap_data[i, j] = 0.99 - abs(i-j)*0.02
            else:
                heatmap_data[i, j] *= np.exp(-abs(i-j)*0.1)
                
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, cmap="rocket", cbar_kws={'label': 'Transition Probability'})
    plt.title("State Transition Heatmap $\mathbf{\\bar{A}}_t$ (Mamba SSM)", fontsize=14)
    plt.xlabel("Hidden Dimension (t-1)", fontsize=12)
    plt.ylabel("Hidden Dimension (t)", fontsize=12)
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300)
    plt.close()

print("Heatmap generated.")
