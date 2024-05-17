import torch
import wandb
import hydra
from tqdm import tqdm

@hydra.main(config_path="configs/train", config_name="config")
def train(cfg):
    with wandb.init(project="challenge_cheese_dinov2") as logger:  # Update project name here

        # Use wandb.config to get hyperparameters
        lr = wandb.config.learning_rate
        batch_size = wandb.config.batch_size
        optimizer_name = wandb.config.optimizer
        weight_decay = wandb.config.weight_decay
        hidden_size = wandb.config.hidden_size
        num_hidden_layers = wandb.config.num_hidden_layers
        num_attention_heads = wandb.config.num_attention_heads
        mlp_ratio = wandb.config.mlp_ratio
        hidden_dropout_prob = wandb.config.hidden_dropout_prob
        attention_probs_dropout_prob = wandb.config.attention_probs_dropout_prob
        image_size = wandb.config.image_size
        patch_size = wandb.config.patch_size

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Instantiate the model with the new hyperparameters
        model = hydra.utils.instantiate(
            cfg.model.instance,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            mlp_ratio=mlp_ratio,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            image_size=image_size,
            patch_size=patch_size
        ).to(device)
        
        # Adjust the optimizer instantiation to use weight_decay
        if optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

        loss_fn = hydra.utils.instantiate(cfg.loss_fn)
        datamodule = hydra.utils.instantiate(cfg.datamodule, batch_size=batch_size)

        train_loader = datamodule.train_dataloader()
        val_loaders = datamodule.val_dataloader()

        for epoch in tqdm(range(cfg.epochs)):
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
                epoch_num_correct += (
                    (preds.argmax(1) == labels).sum().detach().cpu().numpy()
                )
                num_samples += len(images)
            epoch_loss /= num_samples
            epoch_acc = epoch_num_correct / num_samples
            logger.log(
                {
                    "epoch": epoch,
                    "train_loss_epoch": epoch_loss,
                    "train_acc": epoch_acc,
                }
            )
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
                    epoch_num_correct += (
                        (preds.argmax(1) == labels).sum().detach().cpu().numpy()
                    )
                    num_samples += len(images)
                epoch_loss /= num_samples
                epoch_acc = epoch_num_correct / num_samples
                val_metrics[f"{val_set_name}/loss"] = epoch_loss
                val_metrics[f"{val_set_name}/acc"] = epoch_acc
                val_metrics[f"{val_set_name}/confusion_matrix"] = (
                    wandb.plot.confusion_matrix(
                        y_true=y_true,
                        preds=y_pred,
                        class_names=[
                            datamodule.idx_to_class[i][:10].lower()
                            for i in range(len(datamodule.idx_to_class))
                        ],
                    )
                )

            logger.log(
                {
                    "epoch": epoch,
                    **val_metrics,
                }
            )
        torch.save(model.state_dict(), cfg.checkpoint_path)

if __name__ == "__main__":
    sweep_id = 'polytechnique-rabasse/challenge_cheese/7ep0a1an'  # Use the correct sweep ID
    wandb.agent(sweep_id, function=train, count=20)  # Set the number of sweep runs
