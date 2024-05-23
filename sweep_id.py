import wandb

sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val/loss',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'values': [0.1, 0.01, 0.001, 0.0001]
        },
        'batch_size': {
            'values': [16, 32, 64]
        },
        'optimizer': {
            'values': ['adamw', 'sgd']
        },
        'epochs': {
            'values': [10, 20, 30]
        }
    }
}

sweep_id = wandb.sweep(sweep=sweep_config, project="challenge_cheese")

print(f"Sweep ID: {sweep_id}")