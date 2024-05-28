import torch
import copy
import hydra
from omegaconf import OmegaConf

# Define the function to merge weights
def merge_weights(weight1, weight2, alpha=0.5):
    return alpha * weight1 + (1 - alpha) * weight2

# Initialize Hydra
@hydra.main(config_path="configs/train", config_name="config")  
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the first model
    model1 = hydra.utils.instantiate(cfg.model.instance).to(device)
    checkpoint1 = torch.load(cfg.checkpoint_path_1, map_location=device)  
    print(f"Loading model from checkpoint: {cfg.checkpoint_path_1}")

    if 'model_state_dict' in checkpoint1:
        model1.load_state_dict(checkpoint1['model_state_dict'])
    else:
        model1.load_state_dict(checkpoint1)
    model1.eval()

    # Load the second model
    model2 = hydra.utils.instantiate(cfg.model.instance).to(device)
    checkpoint2 = torch.load(cfg.checkpoint_path_2, map_location=device)  
    print(f"Loading model from checkpoint: {cfg.checkpoint_path_2}")

    if 'model_state_dict' in checkpoint2:
        model2.load_state_dict(checkpoint2['model_state_dict'])
    else:
        model2.load_state_dict(checkpoint2)
    model2.eval()

    # Ensure both models have the same architecture
    assert model1.state_dict().keys() == model2.state_dict().keys(), "Models do not have the same architecture"

    # Create a new model instance (same architecture as model1 and model2)
    new_model = hydra.utils.instantiate(cfg.model.instance).to(device)

    # Merge the weights
    new_state_dict = copy.deepcopy(model1.state_dict())
    for param_tensor in model1.state_dict():
        new_state_dict[param_tensor].copy_(
            merge_weights(model1.state_dict()[param_tensor], model2.state_dict()[param_tensor])
        )

    # Load the merged weights into the new model
    new_model.load_state_dict(new_state_dict)
    new_model.eval()

    # Save the new model
    new_model_path = '/users/eleves-a/2022/hippolyte.wallaert/Modal/INF473V-challenge/checkpoints/merged_model_80.pt'  # Specify the path to save the new model
    torch.save(new_model.state_dict(), new_model_path)

    print(f"New model with merged weights saved to: {new_model_path}")

if __name__ == "__main__":
    main()
