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
from utils.arguments import parse_args
from utils.plots import plot_predictions, plot_training_comparison
from utils.scripts import train_model


def main():
    """
    Main training function that:
    1. Sets up device and generates data
    2. Trains both standard and physics-informed models
    3. Compares and evaluates model performances
    """
    # Set random seeds for reproducibility
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(SEED)

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
        physics_informed=False,
        lambda_phy=args.lambda_phy,
        grad_clip_value=args.grad_clip_value,
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
        lambda_phy=args.lambda_phy,
        grad_clip_value=args.grad_clip_value,
    )

    # Compare training histories
    print("\nComparing model performances...")
    plot_training_comparison(standard_history, physics_history)
    print("\nComparison results saved in 'results_comparison' directory")

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
