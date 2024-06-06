import torch
import wandb
import hydra
from tqdm import tqdm

####
# Add the following line in configs
#init_model_weights_path: 


def load_and_modify_weights(model, init_weights_path):
    # Load the initial weights from the given path
    init_state_dict = torch.load(init_weights_path)

    # Load the modified weights into the model
    model.load_state_dict(init_state_dict)

@hydra.main(config_path="configs/train", config_name="config")
def train(cfg):
    # Initialize wandb
    logger = wandb.init(project="challenge_cheese", name=cfg.experiment_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = hydra.utils.instantiate(cfg.model.instance).to(device)

    # Load and modify initial weights from the init model
    init_weights_path = cfg.init_model_weights_path  # Path provided in config
    load_and_modify_weights(model, init_weights_path)

    # Instantiate optimizer, loss function, and data module
    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    # Get data loaders
    train_loader = datamodule.train_dataloader()
    val_loaders = datamodule.val_dataloader()

    for epoch in tqdm(range(cfg.epochs)):
        model.train()
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0

        for i, batch in enumerate(train_loader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)
            loss = loss_fn(preds, labels)
            logger.log({"loss": loss.item()})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(images)
            epoch_num_correct += (preds.argmax(1) == labels).sum().item()
            num_samples += len(images)

        epoch_loss /= num_samples
        epoch_acc = epoch_num_correct / num_samples
        logger.log({
            "epoch": epoch,
            "train_loss_epoch": epoch_loss,
            "train_acc": epoch_acc,
        })

        model.eval()
        val_metrics = {}
        with torch.no_grad():
            for val_set_name, val_loader in val_loaders.items():
                val_epoch_loss = 0
                val_epoch_num_correct = 0
                val_num_samples = 0
                y_true = []
                y_pred = []

                for i, batch in enumerate(val_loader):
                    images, labels = batch
                    images = images.to(device)
                    labels = labels.to(device)
                    preds = model(images)
                    loss = loss_fn(preds, labels)
                    y_true.extend(labels.cpu().tolist())
                    y_pred.extend(preds.argmax(1).cpu().tolist())
                    val_epoch_loss += loss.item() * len(images)
                    val_epoch_num_correct += (preds.argmax(1) == labels).sum().item()
                    val_num_samples += len(images)

                val_epoch_loss /= val_num_samples
                val_epoch_acc = val_epoch_num_correct / val_num_samples
                val_metrics[f"{val_set_name}/loss"] = val_epoch_loss
                val_metrics[f"{val_set_name}/acc"] = val_epoch_acc
                val_metrics[f"{val_set_name}/confusion_matrix"] = wandb.plot.confusion_matrix(
                    y_true=y_true,
                    preds=y_pred,
                    class_names=[datamodule.idx_to_class[i][:10].lower() for i in range(len(datamodule.idx_to_class))]
                )

        logger.log({
            "epoch": epoch,
            **val_metrics,
        })

    # Save the final model
    torch.save(model.state_dict(), cfg.checkpoint_path)

if __name__ == "__main__":
    train()
