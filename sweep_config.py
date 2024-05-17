import wandb

sweep_config = {
    'method': 'random',  # 'grid', 'random', or 'bayes'
    'metric': {
        'name': 'val/acc',  # The metric to optimize
        'goal': 'maximize'  # 'minimize' or 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'values': [1e-4, 1e-5, 1e-6]
        },
        'batch_size': {
            'values': [32, 64, 128]
        },
        'optimizer': {
            'values': ['adamw', 'sgd']
        },
        'weight_decay': {
            'values': [0.01, 0.001, 0.0001]
        },
        'hidden_size': {
            'values': [768, 1024, 2048]
        },
        'num_hidden_layers': {
            'values': [8, 12, 16]
        },
        'num_attention_heads': {
            'values': [8, 12, 16]
        },
        'mlp_ratio': {
            'values': [2, 4, 6]
        },
        'hidden_dropout_prob': {
            'values': [0.0, 0.1, 0.3]
        },
        'attention_probs_dropout_prob': {
            'values': [0.0, 0.1, 0.3]
        },
        'image_size': {
            'values': [224, 256, 384]
        },
        'patch_size': {
            'values': [16, 32]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project='challenge_cheese')
print(f"Sweep ID: {sweep_id}")
