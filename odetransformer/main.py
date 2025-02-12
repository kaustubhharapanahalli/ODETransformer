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

NUM_SAMPLES = 10000
T = 10
NUM_POINTS = 1000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
GRAD_CLIP_VALUE = 1.0


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_training_history(history, save_dir):
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
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Create save directory for checkpoints and plots
    save_dir = "results"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Generate data
    print("Generating vehicle motion data...")
    contexts, time_seq, states = generate_vehicle_motion_data(
        num_samples=NUM_SAMPLES, T=T, num_points=NUM_POINTS
    )
    print("Shape of contexts: ", contexts.shape)
    print("Shape of time_seq: ", time_seq.shape)
    print("Shape of states: ", states.shape)

    # Normalize data
    context_mean = np.mean(contexts, axis=0, keepdims=True)
    context_std = np.std(contexts, axis=0, keepdims=True) + 1e-6
    state_mean = np.mean(states, axis=(0, 1), keepdims=True)
    state_std = np.std(states, axis=(0, 1), keepdims=True) + 1e-6

    print("Context mean: ", context_mean)
    print("Context std: ", context_std)
    print("State mean: ", state_mean)
    print("State std: ", state_std)

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

    print("Shape of train_contexts: ", train_contexts.shape)
    print("Shape of val_contexts: ", val_contexts.shape)
    print("Shape of test_contexts: ", test_contexts.shape)

    train_states = states[train_indices]
    val_states = states[val_indices]
    test_states = states[test_indices]

    print("Shape of train_states: ", train_states.shape)
    print("Shape of val_states: ", val_states.shape)
    print("Shape of test_states: ", test_states.shape)

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
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    # Initialize model, loss, optimizer
    model = VehicleMotionTransformer().to(device)

    # Print model summary
    print("\nModel Summary:")
    summary(
        model,
        input_data=[
            torch.zeros(32, 1000, 1).to(device),
            torch.zeros(32, 5).to(device),
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
    criterion = nn.MSELoss()

    print("Training Transformer for Vehicle Motion...")

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

            optimizer.zero_grad()
            pred = model(t_seq, context)
            loss = criterion(pred, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_VALUE)
            optimizer.step()

            total_train_loss += loss.item()
            batch_pbar.set_postfix({"batch_loss": f"{loss.item():.6f}"})

        avg_train_loss = total_train_loss / len(train_dataloader)

        # Validation phase
        avg_val_loss = validate(model, val_dataloader, criterion, device)

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

        if (epoch + 1) % 5 == 0:
            # Save checkpoint based on validation loss
            if avg_val_loss < history["best_val_loss"]:
                history["best_val_loss"] = avg_val_loss
                history["best_loss"] = save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    avg_val_loss,
                    history["best_loss"],
                    save_dir,
                )

            # Save training history and plot
            with open(f"{save_dir}/training_history.json", "w") as f:
                json.dump(history, f, indent=4)
            plot_training_history(history, save_dir)

    print("\nEvaluating model on a sample...")
    model.eval()
    with torch.no_grad():
        t_seq_sample, context_sample, target_sample = test_dataset[0]
        t_seq_sample = t_seq_sample.unsqueeze(0).to(device)
        context_sample = context_sample.unsqueeze(0).to(device)
        pred_sample = model(t_seq_sample, context_sample)

        # Plot predictions vs ground truth (denormalized)
        plot_predictions(
            pred_sample.squeeze(0).cpu().numpy(),
            target_sample.cpu().numpy(),
            time_seq,
            save_dir,
            state_mean.squeeze(),  # Remove extra dimensions
            state_std.squeeze(),  # Remove extra dimensions
        )

    print("\nTraining completed! Check the 'results' directory for:")
    print("- Model checkpoints (best_model.pt, latest_model.pt)")
    print("- Training history plot (training_loss.png)")
    print("- Prediction vs Ground Truth plot (predictions.png)")


if __name__ == "__main__":
    main()
