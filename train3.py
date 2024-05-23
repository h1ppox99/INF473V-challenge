import torch
import wandb
import hydra
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

# Sweep configuration
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'real_val/acc',
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
            'values': ['AdamW']
        },
        'epochs': {
            'values': [10, 20, 30]
        }
    }
}

sweep_id = wandb.sweep(sweep=sweep_config, project="challenge_cheese")

@hydra.main(config_path="configs/train", config_name="config")
def train(cfg: DictConfig):
    with wandb.init() as logger:
        cfg = OmegaConf.to_container(cfg, resolve=True)

        # Overwrite hyperparameters with sweep parameters
        cfg['learning_rate'] = wandb.config.learning_rate
        cfg['batch_size'] = wandb.config.batch_size
        cfg['epochs'] = wandb.config.epochs
        cfg['datamodule']['batch_size'] = wandb.config.batch_size
        # Update optimizer configuration
        optimizer_target = {
            'AdamW': 'torch.optim.AdamW',
            'sgd': 'torch.optim.SGD'
        }
        cfg['optim']['lr'] = wandb.config.learning_rate
        cfg['optim']['_target_'] = optimizer_target[wandb.config.optimizer]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = hydra.utils.instantiate(cfg['model']['instance']).to(device)
        optimizer = hydra.utils.instantiate(cfg['optim'], params=model.parameters())
        loss_fn = hydra.utils.instantiate(cfg['loss_fn'])
        datamodule = hydra.utils.instantiate(cfg['datamodule'])

        train_loader = datamodule.train_dataloader()
        val_loaders = datamodule.val_dataloader()

        for epoch in tqdm(range(cfg['epochs'])):
            epoch_loss = 0
            epoch_num_correct = 0
            num_samples = 0
            for i, batch in enumerate(train_loader):
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                preds = model(images)
                loss = loss_fn(preds, labels)
                logger.log({"loss": loss.detach().cpu().numpy()})
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().cpu().numpy() * len(images)
                epoch_num_correct += (preds.argmax(1) == labels).sum().detach().cpu().numpy()
                num_samples += len(images)
            epoch_loss /= num_samples
            epoch_acc = epoch_num_correct / num_samples
            logger.log({"epoch": epoch, "train_loss_epoch": epoch_loss, "train_acc": epoch_acc})

            val_metrics = {}
            for val_set_name, val_loader in val_loaders.items():
                epoch_loss = 0
                epoch_num_correct = 0
                num_samples = 0
                y_true = []
                y_pred = []
                for i, batch in enumerate(val_loader):
                    images, labels = batch
                    images = images.to(device)
                    labels = labels.to(device)
                    preds = model(images)
                    loss = loss_fn(preds, labels)
                    y_true.extend(labels.detach().cpu().tolist())
                    y_pred.extend(preds.argmax(1).detach().cpu().tolist())
                    epoch_loss += loss.detach().cpu().numpy() * len(images)
                    epoch_num_correct += (preds.argmax(1) == labels).sum().detach().cpu().numpy()
                    num_samples += len(images)
                epoch_loss /= num_samples
                epoch_acc = epoch_num_correct / num_samples
                val_metrics[f"{val_set_name}/loss"] = epoch_loss
                val_metrics[f"{val_set_name}/acc"] = epoch_acc
                val_metrics[f"{val_set_name}/confusion_matrix"] = wandb.plot.confusion_matrix(
                    y_true=y_true,
                    preds=y_pred,
                    class_names=[datamodule.idx_to_class[i][:10].lower() for i in range(len(datamodule.idx_to_class))],
                )

            logger.log({"epoch": epoch, **val_metrics})
        torch.save(model.state_dict(), cfg['checkpoint_path'])

if __name__ == "__main__":
    wandb.agent(sweep_id, function=train, count=20)