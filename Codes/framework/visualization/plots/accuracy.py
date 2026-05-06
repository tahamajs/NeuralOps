import matplotlib.pyplot as plt
from pathlib import Path

def plot_accuracy_curve(history, metrics, config, output_dir, opts):
    steps = [h.get('step', i) for i, h in enumerate(history) if 'eval_accuracy' in h]
    acc = [h['eval_accuracy'] for h in history if 'eval_accuracy' in h]
    if not steps:
        return []
    fig, ax = plt.subplots(figsize=opts.get('figsize', [10, 6]))
    ax.plot(steps, acc, 'o-', color='#5DC963', markersize=4)
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Accuracy')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = Path(output_dir) / 'accuracy_curve.png'
    fig.savefig(path, dpi=opts.get('dpi', 150), bbox_inches='tight')
    plt.close(fig)
    return [str(path)]