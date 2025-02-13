"""Utility functions for model training and visualization."""

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


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
