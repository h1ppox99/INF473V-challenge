import torch
import wandb
import hydra
from tqdm import tqdm


@hydra.main(config_path="configs/train", config_name="config")
def train(cfg):
    logger = wandb.init(project="challenge_cheese", name=cfg.experiment_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    train_loader = datamodule.train_dataloader()
    val_loaders = datamodule.val_dataloader()
    log_interval = 10  # log every 10 batches

    for epoch in tqdm(range(cfg.epochs)):
        model.train()
        epoch_loss = 0.0
        epoch_num_correct = 0
        num_samples = 0
        for i, batch in enumerate(train_loader):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()

            if i % log_interval == 0:
                logger.log({"batch_loss": loss.item()})

            epoch_loss += loss.item() * len(images)
            epoch_num_correct += (preds.argmax(1) == labels).sum().item()
            num_samples += len(images)

        epoch_loss /= num_samples
        epoch_acc = epoch_num_correct / num_samples
        logger.log({"epoch_loss": epoch_loss, "epoch_accuracy": epoch_acc})

        # Validation step
        validate(model, device, val_loaders, loss_fn, logger, datamodule.idx_to_class, epoch)

    torch.save(model.state_dict(), cfg.checkpoint_path)

def validate(model, device, val_loaders, loss_fn, logger, idx_to_class, epoch):
    model.eval()
    with torch.no_grad():
        for val_set_name, val_loader in val_loaders.items():
            val_loss, val_acc, num_samples = 0.0, 0.0, 0
            y_true, y_pred = [], []
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images)
                loss = loss_fn(preds, labels)
                val_loss += loss.item() * len(images)
                val_acc += (preds.argmax(1) == labels).sum().item()
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(preds.argmax(1).cpu().tolist())
                num_samples += len(images)

            val_loss /= num_samples
            val_acc /= num_samples
            logger.log({f"{val_set_name}/loss": val_loss, f"{val_set_name}/accuracy": val_acc,
                        f"{val_set_name}/confusion_matrix": wandb.plot.confusion_matrix(
                            y_true, y_pred, class_names=[idx_to_class[i][:10].lower() for i in range(len(idx_to_class))])})

if __name__ == "__main__":
    train()