"""`train` command."""
def cmd_train(args):
    print(f"Training using config: {args.config}")
    from neuralops.trainer.trainer import create_trainer
    print("Trainer factory loaded.")
    return 0
