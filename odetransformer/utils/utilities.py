import json
from datetime import datetime
from pathlib import Path

import torch


def count_parameters(model):
    """Count number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, epoch, loss, best_loss, save_dir):
    """
    Save model checkpoint and return updated best loss.

    Args:
        model: The neural network model
        optimizer: The optimizer
        epoch (int): Current epoch number
        loss (float): Current loss value
        best_loss (float): Best loss value so far
        save_dir (str): Directory to save checkpoints

    Returns:
        float: Updated best loss value
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "best_loss": best_loss,
    }
    torch.save(checkpoint, f"{save_dir}/latest_model.pt")
    if loss < best_loss:
        torch.save(checkpoint, f"{save_dir}/best_model.pt")
        return loss
    return best_loss


def create_experiment_folder(
    model_type, hyperparameters, physics_informed=False
):
    """
    Creates a uniquely named experiment folder with configuration details.

    Args:
        model_type (str): Type of model being trained
        hyperparameters (dict): Dictionary of hyperparameters
        physics_informed (bool): Whether using physics-informed approach

    Returns:
        str: Path to the created experiment folder
    """
    # Create base experiments directory
    base_dir = Path("experiments")
    base_dir.mkdir(exist_ok=True)

    # Generate timestamp-based folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_suffix = "_physics" if physics_informed else "_standard"
    exp_dir = base_dir / f"{timestamp}" / f"{model_type}"
    exp_dir.mkdir(parents=True)

    # Create experiment configuration
    config = {
        "timestamp": timestamp,
        "model_type": model_type,
        "physics_informed": physics_informed,
        "hyperparameters": hyperparameters,
        "model_architecture": {
            "d_model": hyperparameters.get("d_model", 8),
            "num_heads": hyperparameters.get("num_heads", 2),
            "num_layers": hyperparameters.get("num_layers", 2),
        },
    }

    # Save configuration
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # Create subdirectories
    (exp_dir / "checkpoints").mkdir()
    (exp_dir / "plots").mkdir()
    (exp_dir / "metrics").mkdir()

    return str(exp_dir)
