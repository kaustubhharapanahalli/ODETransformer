"""
Combined training script for standard and physics-informed vehicle motion prediction.

This module implements training of two transformer models to predict vehicle motion:
1. A standard model trained only on data
2. A physics-informed model trained with both data and physics-based loss terms

Key components:
- Data generation and preprocessing with normalization
- Model training loops for both standard and physics-informed approaches
- Validation and model checkpointing
- Visualization of training progress and model comparisons
- Evaluation on test data

The physics-informed model enforces physical constraints:
- dx/dt = v (velocity is derivative of position)
- dv/dt = a (acceleration is derivative of velocity)
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from architecture.vehicle_motion_transformer import VehicleMotionTransformer
from dataset.generator import (
    VehicleMotionDataset,
    generate_vehicle_motion_data,
)
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

# Training hyperparameters
NUM_SAMPLES = 10000  # Total number of trajectories to generate
T = 10  # Time duration in seconds
NUM_POINTS = 1000  # Number of timesteps per trajectory
BATCH_SIZE = 32  # Batch size for training
LEARNING_RATE = 1e-4  # Learning rate for optimizer
NUM_EPOCHS = 20  # Number of training epochs
GRAD_CLIP_VALUE = 1.0  # Maximum gradient norm for clipping
LAMBDA_PHY = 0.1  # Weight for physics-informed loss term


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
    plt.savefig(f"{save_dir}/training_loss.png")
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
    plt.savefig(f"{save_dir}/predictions.png")
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


def validate(model, val_dataloader, criterion, device, physics_informed=False):
    """
    Validate model on validation dataset.

    Args:
        model: The neural network model
        val_dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to run validation on
        physics_informed (bool): Whether using physics-informed approach

    Returns:
        float: Average validation loss
    """
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for t_seq, context, target in val_dataloader:
            t_seq = t_seq.to(device)
            context = context.to(device)
            target = target.to(device)

            pred = model(t_seq, context)
            if physics_informed:
                loss = criterion(
                    pred, target
                )  # Only use data loss for validation
            else:
                loss = criterion(pred, target)
            total_val_loss += loss.item()

    return total_val_loss / len(val_dataloader)


def train_epoch(
    model,
    train_dataloader,
    optimizer,
    criterion,
    device,
    physics_informed=False,
):
    """
    Train model for one epoch.

    Args:
        model: The neural network model
        train_dataloader: DataLoader for training data
        optimizer: The optimizer
        criterion: Loss function
        device: Device to run training on
        physics_informed (bool): Whether to use physics-informed approach

    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_train_loss = 0.0

    batch_pbar = tqdm(
        train_dataloader,
        desc="Training batch",
        leave=False,
    )

    for t_seq, context, target in batch_pbar:
        t_seq = t_seq.to(device)
        context = context.to(device)
        target = target.to(device)

        if physics_informed:
            t_seq.requires_grad_(True)

        optimizer.zero_grad()
        pred = model(t_seq, context)

        data_loss = criterion(pred, target)

        if physics_informed:
            # Physics-informed loss computation
            x_pred, v_pred, a_pred = pred[..., 0], pred[..., 1], pred[..., 2]
            B, T_steps = x_pred.shape
            t_flat = t_seq.view(-1)
            x_flat, v_flat = x_pred.view(-1), v_pred.view(-1)

            # Compute gradients for physics constraints
            grad_x = torch.autograd.grad(
                x_flat,
                t_flat,
                grad_outputs=torch.ones_like(x_flat),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            grad_v = torch.autograd.grad(
                v_flat,
                t_flat,
                grad_outputs=torch.ones_like(v_flat),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]

            grad_x = torch.zeros_like(x_flat) if grad_x is None else grad_x
            grad_v = torch.zeros_like(v_flat) if grad_v is None else grad_v

            grad_x = grad_x.view(B, T_steps)
            grad_v = grad_v.view(B, T_steps)

            # Physics residuals
            res_x = grad_x - v_pred  # dx/dt = v
            res_v = grad_v - a_pred  # dv/dt = a
            physics_loss = res_x.pow(2).mean() + res_v.pow(2).mean()
            loss = data_loss + LAMBDA_PHY * physics_loss
        else:
            loss = data_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_VALUE)
        optimizer.step()

        total_train_loss += loss.item()
        batch_pbar.set_postfix({"batch_loss": f"{loss.item():.6f}"})

    return total_train_loss / len(train_dataloader)


def train_model(
    model_type,
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    criterion,
    device,
    save_dir,
    physics_informed=False,
):
    """
    Train model for multiple epochs.

    Args:
        model_type (str): Type of model being trained
        model: The neural network model
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        optimizer: The optimizer
        criterion: Loss function
        device: Device to run training on
        save_dir (str): Directory to save checkpoints and plots
        physics_informed (bool): Whether to use physics-informed approach

    Returns:
        tuple: Training history and trained model
    """
    history = {
        "train_loss": [],
        "val_loss": [],
        "best_loss": float("inf"),
        "best_val_loss": float("inf"),
    }

    epoch_pbar = tqdm(
        range(NUM_EPOCHS), desc=f"{model_type} Training Progress"
    )

    for epoch in epoch_pbar:
        avg_train_loss = train_epoch(
            model,
            train_dataloader,
            optimizer,
            criterion,
            device,
            physics_informed,
        )

        avg_val_loss = validate(
            model, val_dataloader, criterion, device, physics_informed
        )

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        epoch_pbar.set_postfix(
            {
                "train_loss": f"{avg_train_loss:.6f}",
                "val_loss": f"{avg_val_loss:.6f}",
            }
        )

        if (epoch + 1) % 5 == 0:
            history["best_loss"] = save_checkpoint(
                model,
                optimizer,
                epoch,
                avg_train_loss,
                history["best_loss"],
                save_dir,
            )

            with open(f"{save_dir}/training_history.json", "w") as f:
                json.dump(history, f, indent=4)
            plot_training_history(history, save_dir, model_type)

    return history, model


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
        "standard_model": {
            "final_train_loss": standard_history["train_loss"][-1],
            "final_val_loss": standard_history["val_loss"][-1],
            "best_val_loss": standard_history["best_val_loss"],
        },
        "physics_informed_model": {
            "final_train_loss": physics_history["train_loss"][-1],
            "final_val_loss": physics_history["val_loss"][-1],
            "best_val_loss": physics_history["best_val_loss"],
        },
    }

    with open(f"{save_dir}/comparison_metrics.json", "w") as f:
        json.dump(comparison_metrics, f, indent=4)


def main():
    """
    Main training function that:
    1. Sets up device and generates data
    2. Trains both standard and physics-informed models
    3. Compares and evaluates model performances
    """
    # Set device
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Generate and prepare data (using the data preparation code from main.py)
    print("Generating vehicle motion data...")
    contexts, time_seq, states = generate_vehicle_motion_data(
        num_samples=NUM_SAMPLES, T=T, num_points=NUM_POINTS
    )

    # Data normalization and splitting code
    context_mean = np.mean(contexts, axis=0, keepdims=True)
    context_std = np.std(contexts, axis=0, keepdims=True) + 1e-6
    state_mean = np.mean(states, axis=(0, 1), keepdims=True)
    state_std = np.std(states, axis=(0, 1), keepdims=True) + 1e-6

    # Split indices
    indices = np.arange(NUM_SAMPLES)
    np.random.shuffle(indices)
    train_end = int(NUM_SAMPLES * 0.8)
    val_end = int(NUM_SAMPLES * 0.9)

    # Create datasets and dataloaders
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    # Prepare data splits
    data_splits = {
        "train": (contexts[train_indices], states[train_indices]),
        "val": (contexts[val_indices], states[val_indices]),
        "test": (contexts[test_indices], states[test_indices]),
    }

    # Create datasets
    datasets = {
        split: VehicleMotionDataset(
            data_splits[split][0],
            time_seq,
            data_splits[split][1],
            context_mean,
            context_std,
            state_mean,
            state_std,
        )
        for split in ["train", "val", "test"]
    }

    # Create dataloaders
    dataloaders = {
        "train": DataLoader(
            datasets["train"], batch_size=BATCH_SIZE, shuffle=True
        ),
        "val": DataLoader(
            datasets["val"], batch_size=BATCH_SIZE, shuffle=False
        ),
        "test": DataLoader(
            datasets["test"], batch_size=BATCH_SIZE, shuffle=False
        ),
    }

    # Train standard model
    print("\nTraining Standard Transformer Model...")
    standard_model = VehicleMotionTransformer().to(device)
    standard_optimizer = optim.AdamW(
        standard_model.parameters(), lr=LEARNING_RATE
    )
    standard_criterion = nn.MSELoss()

    standard_history, standard_model = train_model(
        "Standard",
        standard_model,
        dataloaders["train"],
        dataloaders["val"],
        standard_optimizer,
        standard_criterion,
        device,
        "results_standard",
    )

    # Train physics-informed model
    print("\nTraining Physics-Informed Transformer Model...")
    physics_model = VehicleMotionTransformer().to(device)
    physics_optimizer = optim.AdamW(
        physics_model.parameters(), lr=LEARNING_RATE
    )
    physics_criterion = nn.MSELoss()

    physics_history, physics_model = train_model(
        "Physics-Informed",
        physics_model,
        dataloaders["train"],
        dataloaders["val"],
        physics_optimizer,
        physics_criterion,
        device,
        "results_physics",
        physics_informed=True,
    )

    # Compare training histories
    print("\nComparing model performances...")
    plot_training_comparison(standard_history, physics_history)
    print("\nComparison results saved in 'results_comparison' directory")

    # Evaluate both models on test sample
    for model_type, model in [
        ("Standard", standard_model),
        ("Physics-Informed", physics_model),
    ]:
        print(f"\nEvaluating {model_type} model on a sample...")
        save_dir = f"results_{model_type.lower()}"

        model.eval()
        with torch.no_grad():
            t_seq_sample, context_sample, target_sample = datasets["test"][0]
            t_seq_sample = t_seq_sample.unsqueeze(0).to(device)
            context_sample = context_sample.unsqueeze(0).to(device)
            pred_sample = model(t_seq_sample, context_sample)

            plot_predictions(
                pred_sample.squeeze(0).cpu().numpy(),
                target_sample.cpu().numpy(),
                time_seq,
                save_dir,
                state_mean.squeeze(),
                state_std.squeeze(),
            )

    print("\nTraining completed! Check the results directories for outputs.")


if __name__ == "__main__":
    main()
