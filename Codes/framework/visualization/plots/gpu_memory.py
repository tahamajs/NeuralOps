import matplotlib.pyplot as plt

def plot_gpu_memory(history, metrics, config, output_dir, opts):
    steps_gpu = [h.get('step', i) for i, h in enumerate(history) if 'gpu_mem' in h]
    mem = [h['gpu_mem'] for h in history if 'gpu_mem' in h]
    steps_cpu = [h.get('step', i) for i, h in enumerate(history) if 'cpu_mem' in h]
    cpu_mem = [h['cpu_mem'] for h in history if 'cpu_mem' in h]
    if not steps_gpu and not steps_cpu:
        return []
    fig, ax = plt.subplots(figsize=opts.get("figsize", [10, 4]))
    if steps_gpu:
        ax.plot(steps_gpu, mem, label='GPU Memory (MB)', color='#3B528B')
    if steps_cpu:
        ax.plot(steps_cpu, cpu_mem, label='CPU Memory (MB)', color='#5DC963')
    ax.set_xlabel('Step')
    ax.set_ylabel('Memory (MB)')
    ax.set_title('Memory Usage')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "gpu_memory.png"
    fig.savefig(path, dpi=opts.get("dpi", 150), bbox_inches='tight')
    plt.close(fig)
    return [str(path)]