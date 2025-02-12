"""
Physics-informed neural network training for vehicle motion prediction.

This module implements training of a transformer model to predict vehicle motion
with physics-informed loss terms. The model learns to predict displacement, velocity
and acceleration while respecting the underlying physics equations.

Key components:
- Data generation and preprocessing
- Model training with combined data and physics losses
- Validation and model checkpointing
- Visualization of training progress and predictions

The physics loss enforces:
dx/dt = v (velocity is derivative of position)
dv/dt = a (acceleration is derivative of velocity)
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


def count_parameters(model):
    """Count number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_training_history(history, save_dir):
    """
    Plot and save training and validation loss curves.

    Args:
        history (dict): Dictionary containing training metrics
        save_dir (str): Directory to save the plot
    """
    # Create save directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss Over Time")
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
    # Create save directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Denormalize predictions and target if normalization parameters are provided
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


def validate(model, val_dataloader, criterion, device):
    """
    Validate model on validation dataset.

    Args:
        model: The neural network model
        val_dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to run validation on

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
            loss = criterion(pred, target)
            total_val_loss += loss.item()

    return total_val_loss / len(val_dataloader)


def main():
    """
    Main training function that:
    1. Sets up device and directories
    2. Generates and preprocesses data
    3. Creates and trains the model
    4. Evaluates and saves results
    """
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Create save directory for checkpoints and plots
    save_dir = "results_physics"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Generate data
    print("Generating vehicle motion data...")
    contexts, time_seq, states = generate_vehicle_motion_data(
        num_samples=NUM_SAMPLES, T=T, num_points=NUM_POINTS
    )

    # Normalize data
    context_mean = np.mean(contexts, axis=0, keepdims=True)
    context_std = np.std(contexts, axis=0, keepdims=True) + 1e-6
    state_mean = np.mean(states, axis=(0, 1), keepdims=True)
    state_std = np.std(states, axis=(0, 1), keepdims=True) + 1e-6

    # Split data into train/val/test sets
    indices = np.arange(NUM_SAMPLES)
    np.random.shuffle(indices)
    train_end = int(NUM_SAMPLES * 0.8)
    val_end = int(NUM_SAMPLES * 0.9)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_contexts = contexts[train_indices]
    val_contexts = contexts[val_indices]
    test_contexts = contexts[test_indices]

    train_states = states[train_indices]
    val_states = states[val_indices]
    test_states = states[test_indices]

    # Create datasets with normalization
    train_dataset = VehicleMotionDataset(
        train_contexts,
        time_seq,
        train_states,
        context_mean,
        context_std,
        state_mean,
        state_std,
    )
    val_dataset = VehicleMotionDataset(
        val_contexts,
        time_seq,
        val_states,
        context_mean,
        context_std,
        state_mean,
        state_std,
    )
    test_dataset = VehicleMotionDataset(
        test_contexts,
        time_seq,
        test_states,
        context_mean,
        context_std,
        state_mean,
        state_std,
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    model = VehicleMotionTransformer().to(device)

    # Print model summary
    print("\nModel Summary:")
    summary(
        model,
        input_data=[
            torch.zeros(BATCH_SIZE, NUM_POINTS, 1).to(device),
            torch.zeros(BATCH_SIZE, 5).to(device),
        ],
        depth=4,
        col_names=[
            "input_size",
            "output_size",
            "num_params",
            "kernel_size",
            "mult_adds",
        ],
        col_width=20,
        row_settings=["var_names"],
    )

    print(f"\nTotal trainable parameters: {count_parameters(model):,}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    data_criterion = nn.MSELoss()

    lambda_phy = 0.1  # Weight for the physics-informed loss
    print("Training Physics-Informed Transformer for Vehicle Motion...")

    history = {
        "train_loss": [],
        "val_loss": [],
        "best_loss": float("inf"),
        "best_val_loss": float("inf"),
    }
    epoch_pbar = tqdm(range(NUM_EPOCHS), desc="Training Progress")

    for epoch in epoch_pbar:
        # Training phase
        model.train()
        total_train_loss = 0.0

        batch_pbar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",
            leave=False,
        )

        for t_seq, context, target in batch_pbar:
            t_seq = t_seq.to(device)
            context = context.to(device)
            target = target.to(device)

            t_seq.requires_grad_(
                True
            )  # Enable gradient computation for physics loss

            optimizer.zero_grad()
            pred = model(t_seq, context)

            # Standard MSE loss on predictions
            data_loss = data_criterion(pred, target)

            # Physics-informed loss computation
            x_pred = pred[..., 0]  # Displacement predictions
            v_pred = pred[..., 1]  # Velocity predictions
            a_pred = pred[..., 2]  # Acceleration predictions

            B, T_steps = x_pred.shape
            t_flat = t_seq.view(-1)
            x_flat = x_pred.view(-1)
            v_flat = v_pred.view(-1)

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

            # Handle potential None gradients
            if grad_x is None:
                grad_x = torch.zeros_like(x_flat)
            if grad_v is None:
                grad_v = torch.zeros_like(v_flat)

            grad_x = grad_x.view(B, T_steps)
            grad_v = grad_v.view(B, T_steps)

            # Physics residuals: dx/dt = v, dv/dt = a
            res_x = grad_x - v_pred
            res_v = grad_v - a_pred
            physics_loss = res_x.pow(2).mean() + res_v.pow(2).mean()

            # Combined loss
            loss = data_loss + lambda_phy * physics_loss
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_VALUE)
            optimizer.step()

            total_train_loss += loss.item()
            batch_pbar.set_postfix({"batch_loss": f"{loss.item():.6f}"})

        avg_train_loss = total_train_loss / len(train_dataloader)

        # Validation phase
        avg_val_loss = validate(model, val_dataloader, data_criterion, device)

        # Update history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        # Update progress bar
        epoch_pbar.set_postfix(
            {
                "train_loss": f"{avg_train_loss:.6f}",
                "val_loss": f"{avg_val_loss:.6f}",
            }
        )

        # Save checkpoints and plots periodically
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
            plot_training_history(history, save_dir)

    # Final evaluation
    print("\nEvaluating model on a sample...")
    model.eval()
    with torch.no_grad():
        t_seq_sample, context_sample, target_sample = test_dataset[0]
        t_seq_sample = t_seq_sample.unsqueeze(0).to(device)
        context_sample = context_sample.unsqueeze(0).to(device)
        pred_sample = model(t_seq_sample, context_sample)

        plot_predictions(
            pred_sample.squeeze(0).cpu().numpy(),
            target_sample.cpu().numpy(),
            time_seq,
            save_dir,
            state_mean.squeeze(),  # Remove extra dimensions
            state_std.squeeze(),  # Remove extra dimensions
        )

    print("\nTraining completed! Check the 'results_physics' directory for:")
    print("- Model checkpoints (best_model.pt, latest_model.pt)")
    print("- Training history plot (training_loss.png)")
    print("- Prediction vs Ground Truth plot (predictions.png)")


if __name__ == "__main__":
    main()
