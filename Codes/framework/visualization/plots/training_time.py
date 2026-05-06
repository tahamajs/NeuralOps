import matplotlib.pyplot as plt

def plot_training_time(history, metrics, config, output_dir, opts):
    times = [h.get('elapsed', 0) for i, h in enumerate(history) if 'elapsed' in h]
    if not times:
        # estimate from step timestamps if available
        steps = [h.get('step', i) for i, h in enumerate(history)]
        times = [i * 0.01 for i in range(len(steps))]   # placeholder
    fig, ax = plt.subplots(figsize=opts.get("figsize", [10, 4]))
    ax.plot(range(len(times)), times, color='#440154')
    ax.set_xlabel('Step')
    ax.set_ylabel('Time (s)')
    ax.set_title('Training Time vs Steps')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "training_time.png"
    fig.savefig(path, dpi=opts.get("dpi", 150), bbox_inches='tight')
    plt.close(fig)
    return [str(path)]