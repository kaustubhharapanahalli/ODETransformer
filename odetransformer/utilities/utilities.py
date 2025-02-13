"""Utility functions for model training and visualization."""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def count_parameters(model):
    """Count number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_training_history(history, save_dir, model_type):
    """
    Plot and save training and validation loss curves.

    Args:
        history (dict): Dictionary containing training metrics
        save_dir (str): Directory to save the plot
        model_type (str): Type of model (Standard or Physics-Informed)
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title(f"{model_type} Training and Validation Loss Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()

    # Update filename to use lowercase and underscore
    model_type_lower = model_type.lower().replace("-", "_")
    plt.savefig(f"{save_dir}/{model_type_lower}_training_loss.png")
    plt.close()


def plot_predictions(
    pred, target, time_seq, save_dir, state_mean=None, state_std=None
):
    """
    Plot and save model predictions against ground truth.

    Args:
        pred (np.ndarray): Model predictions
        target (np.ndarray): Ground truth values
        time_seq (np.ndarray): Time points
        save_dir (str): Directory to save the plot
        state_mean (np.ndarray, optional): Mean for denormalization
        state_std (np.ndarray, optional): Standard deviation for denormalization
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if state_mean is not None and state_std is not None:
        pred = pred * state_std + state_mean
        target = target * state_std + state_mean

    labels = ["Displacement", "Velocity", "Acceleration"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    fig.suptitle("Model Predictions vs Ground Truth")

    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(time_seq, pred[:, i], label="Prediction", alpha=0.8)
        ax.plot(time_seq, target[:, i], label="Ground Truth", alpha=0.8)
        ax.set_xlabel("Time")
        ax.set_ylabel(label)
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    # Get model type from save_dir path
    model_type = "standard" if "standard" in save_dir else "physics_informed"
    plt.savefig(f"{save_dir}/{model_type}_predictions.png")
    plt.close()


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
    exp_dir = base_dir / f"{timestamp}_{model_type}{model_suffix}"
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


def plot_training_comparison(
    standard_history, physics_history, save_dir="results_comparison"
):
    """
    Plot and save comparison of standard and physics-informed model training.

    Args:
        standard_history (dict): Training history for standard model
        physics_history (dict): Training history for physics-informed model
        save_dir (str): Directory to save comparison plots and metrics
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 8))

    # Plot training losses
    plt.subplot(2, 1, 1)
    plt.plot(standard_history["train_loss"], label="Standard Model")
    plt.plot(physics_history["train_loss"], label="Physics-Informed Model")
    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()

    # Plot validation losses
    plt.subplot(2, 1, 2)
    plt.plot(standard_history["val_loss"], label="Standard Model")
    plt.plot(physics_history["val_loss"], label="Physics-Informed Model")
    plt.title("Validation Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/model_comparison.png")
    plt.close()

    # Save comparison metrics
    comparison_metrics = {
        "standard": {
            "final_train_loss": standard_history["train_loss"][-1],
            "final_val_loss": standard_history["val_loss"][-1],
            "best_val_loss": standard_history["best_val_loss"],
        },
        "physics_informed": {
            "final_train_loss": physics_history["train_loss"][-1],
            "final_val_loss": physics_history["val_loss"][-1],
            "best_val_loss": physics_history["best_val_loss"],
        },
    }

    with open(f"{save_dir}/comparison_metrics.json", "w") as f:
        json.dump(comparison_metrics, f, indent=4)


def plot_training_comparison(
    standard_history, physics_history, save_dir="results_comparison"
):
    """
    Plot and save comparison of standard and physics-informed model training.

    Args:
        standard_history (dict): Training history for standard model
        physics_history (dict): Training history for physics-informed model
        save_dir (str): Directory to save comparison plots and metrics
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 8))

    # Plot training losses
    plt.subplot(2, 1, 1)
    plt.plot(standard_history["train_loss"], label="Standard Model")
    plt.plot(physics_history["train_loss"], label="Physics-Informed Model")
    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()

    # Plot validation losses
    plt.subplot(2, 1, 2)
    plt.plot(standard_history["val_loss"], label="Standard Model")
    plt.plot(physics_history["val_loss"], label="Physics-Informed Model")
    plt.title("Validation Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/model_comparison.png")
    plt.close()

    # Save comparison metrics with updated naming
    comparison_metrics = {
        "standard": {
            "final_train_loss": standard_history["train_loss"][-1],
            "final_val_loss": standard_history["val_loss"][-1],
            "best_val_loss": standard_history["best_val_loss"],
        },
        "physics_informed": {
            "final_train_loss": physics_history["train_loss"][-1],
            "final_val_loss": physics_history["val_loss"][-1],
            "best_val_loss": physics_history["best_val_loss"],
        },
    }

    with open(f"{save_dir}/comparison_metrics.json", "w") as f:
        json.dump(comparison_metrics, f, indent=4)
