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

import argparse
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
from utilities.utilities import (
    count_parameters,
    create_experiment_folder,
    plot_predictions,
    plot_training_comparison,
    plot_training_history,
    save_checkpoint,
)

# Training hyperparameters
GRAD_CLIP_VALUE = 1.0  # Maximum gradient norm for clipping
LAMBDA_PHY = 0.1  # Weight for physics-informed loss term


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
    hyperparameters,
    physics_informed=False,
):
    """
    Train model for multiple epochs with experiment tracking.

    Args:
        model_type (str): Type of model being trained
        model: The neural network model
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        optimizer: The optimizer
        criterion: Loss function
        device: Device to run training on
        hyperparameters (dict): Dictionary of training hyperparameters
        physics_informed (bool): Whether to use physics-informed approach

    Returns:
        tuple: Training history, trained model, and experiment directory
    """
    # Create experiment directory
    save_dir = create_experiment_folder(
        model_type, hyperparameters, physics_informed
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "best_loss": float("inf"),
        "best_val_loss": float("inf"),
    }

    epoch_pbar = tqdm(
        range(hyperparameters["NUM_EPOCHS"]),
        desc=f"{model_type} Training Progress",
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
                f"{save_dir}/checkpoints",
            )

            # Save training history
            with open(f"{save_dir}/metrics/training_history.json", "w") as f:
                json.dump(history, f, indent=4)

            # Plot and save training curves
            plot_training_history(history, f"{save_dir}/plots", model_type)

    return history, model, save_dir


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train standard and physics-informed vehicle motion prediction models"
    )

    # Data generation parameters
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Total number of trajectories to generate",
    )
    parser.add_argument(
        "--time-duration",
        type=float,
        default=10,
        help="Time duration in seconds",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=1000,
        help="Number of timesteps per trajectory",
    )

    # Training parameters
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument(
        "--grad-clip-value",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping",
    )
    parser.add_argument(
        "--lambda-phy",
        type=float,
        default=0.1,
        help="Weight for physics-informed loss term",
    )

    # Model architecture parameters
    parser.add_argument(
        "--d-model",
        type=int,
        default=8,
        help="Dimension of model's internal representations",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=2,
        help="Number of attention heads in transformer layers",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of transformer encoder layers",
    )

    return parser.parse_args()


def main():
    """
    Main training function that:
    1. Sets up device and generates data
    2. Trains both standard and physics-informed models
    3. Compares and evaluates model performances
    """
    args = parse_args()

    # Set device
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Generate and prepare data
    print("Generating vehicle motion data...")
    contexts, time_seq, states = generate_vehicle_motion_data(
        num_samples=args.num_samples,
        T=args.time_duration,
        num_points=args.num_points,
    )

    # Data normalization and splitting code
    context_mean = np.mean(contexts, axis=0, keepdims=True)
    context_std = np.std(contexts, axis=0, keepdims=True) + 1e-6
    state_mean = np.mean(states, axis=(0, 1), keepdims=True)
    state_std = np.std(states, axis=(0, 1), keepdims=True) + 1e-6

    # Split indices
    indices = np.arange(args.num_samples)
    np.random.shuffle(indices)
    train_end = int(args.num_samples * 0.8)
    val_end = int(args.num_samples * 0.9)

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
            datasets["train"], batch_size=args.batch_size, shuffle=True
        ),
        "val": DataLoader(
            datasets["val"], batch_size=args.batch_size, shuffle=False
        ),
        "test": DataLoader(
            datasets["test"], batch_size=args.batch_size, shuffle=False
        ),
    }

    hyperparameters = {
        "NUM_SAMPLES": args.num_samples,
        "T": args.time_duration,
        "NUM_POINTS": args.num_points,
        "BATCH_SIZE": args.batch_size,
        "LEARNING_RATE": args.learning_rate,
        "NUM_EPOCHS": args.num_epochs,
        "GRAD_CLIP_VALUE": args.grad_clip_value,
        "LAMBDA_PHY": args.lambda_phy,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
    }

    # Train standard model
    print("\nTraining Standard Transformer Model...")
    standard_model = VehicleMotionTransformer(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    ).to(device)
    standard_optimizer = optim.AdamW(
        standard_model.parameters(), lr=args.learning_rate
    )
    standard_criterion = nn.MSELoss()

    # Define hyperparameters

    standard_history, standard_model, standard_dir = train_model(
        "Standard",
        standard_model,
        dataloaders["train"],
        dataloaders["val"],
        standard_optimizer,
        standard_criterion,
        device,
        hyperparameters,
    )

    # Initialize physics-informed model and training components
    physics_model = VehicleMotionTransformer(
        d_model=hyperparameters["d_model"],
        num_heads=hyperparameters["num_heads"],
        num_layers=hyperparameters["num_layers"],
    ).to(device)
    physics_optimizer = optim.AdamW(
        physics_model.parameters(), lr=hyperparameters["LEARNING_RATE"]
    )
    physics_criterion = nn.MSELoss()

    physics_history, physics_model, physics_dir = train_model(
        "Physics-Informed",
        physics_model,
        dataloaders["train"],
        dataloaders["val"],
        physics_optimizer,
        physics_criterion,
        device,
        hyperparameters,
        physics_informed=True,
    )

    # Compare training histories
    print("\nComparing model performances...")
    plot_training_comparison(standard_history, physics_history)
    print("\nComparison results saved in 'results_comparison' directory")

    # Evaluate both models on test sample
    # Evaluate both models on test sample
    for model_type, model, save_dir in [
        ("Standard", standard_model, standard_dir),
        ("Physics-Informed", physics_model, physics_dir),
    ]:
        print(f"\nEvaluating {model_type} model on a sample...")
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
                f"{save_dir}/plots",
                state_mean.squeeze(),
                state_std.squeeze(),
            )

    print("\nTraining completed! Check the results directories for outputs.")


if __name__ == "__main__":
    main()
