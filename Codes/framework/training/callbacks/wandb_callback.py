class WandbCallback:
    def __init__(self, project_name, run_name, config):
        try:
            import wandb
            self.wandb = wandb
            self.wandb.init(project=project_name, name=run_name, config=config)
        except ImportError:
            print("wandb not installed. Skipping.")
            self.wandb = None

    def log_metrics(self, metrics, step=None):
        if self.wandb:
            self.wandb.log(metrics, step=step)

    def finish(self):
        if self.wandb:
            self.wandb.finish()