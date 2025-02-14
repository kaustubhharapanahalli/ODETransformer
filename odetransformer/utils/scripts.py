import json

import torch
from tqdm import tqdm
from utils.plots import plot_training_history
from utils.utilities import create_experiment_folder, save_checkpoint


def validate(
    model,
    val_dataloader,
    criterion,
    device,
    physics_informed=False,
):
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

    val_pbar = tqdm(
        val_dataloader,
        desc="Validation",
        leave=False,
    )

    with torch.no_grad():
        for t_seq, context, target in val_pbar:
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
            val_pbar.set_postfix({"val_loss": f"{loss.item():.6f}"})

    return total_val_loss / len(val_dataloader)


def train_epoch(
    model,
    train_dataloader,
    optimizer,
    criterion,
    device,
    physics_informed=False,
    lambda_phy=0.1,
    grad_clip_value=1.0,
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
            # Physics-informed loss computation for predictions and ground truth
            x_pred, v_pred, a_pred = pred[..., 0], pred[..., 1], pred[..., 2]
            x_true, v_true, a_true = (
                target[..., 0],
                target[..., 1],
                target[..., 2],
            )
            B, T_steps = x_pred.shape

            # Initialize physics residuals for both predictions and ground truth
            res_x_pred = torch.zeros_like(x_pred)
            res_v_pred = torch.zeros_like(v_pred)
            res_x_true = torch.zeros_like(x_true)
            res_v_true = torch.zeros_like(v_true)

            # Compute gradients for each batch
            for b in range(B):
                # Compute gradients for predictions
                grad_x_pred = torch.autograd.grad(
                    x_pred[b],
                    t_seq[b],
                    grad_outputs=torch.ones_like(x_pred[b]),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )[0]

                grad_v_pred = torch.autograd.grad(
                    v_pred[b],
                    t_seq[b],
                    grad_outputs=torch.ones_like(v_pred[b]),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )[0]

                # Physics residuals for predictions
                grad_x_pred = (
                    torch.zeros_like(x_pred[b])
                    if grad_x_pred is None
                    else grad_x_pred
                )
                grad_v_pred = (
                    torch.zeros_like(v_pred[b])
                    if grad_v_pred is None
                    else grad_v_pred
                )

                # Physics residuals
                res_x_pred[b] = grad_x_pred - v_pred[b]  # dx/dt = v
                res_v_pred[b] = grad_v_pred - a_pred[b]  # dv/dt = a
                res_x_true[b] = v_true[b]  # Known: dx/dt = v
                res_v_true[b] = a_true[b]  # Known: dv/dt = a

            # Compute physics loss comparing prediction residuals with ground truth
            physics_loss = (
                (
                    (res_x_pred - res_x_true).pow(2).mean()
                    + (res_v_pred - res_v_true).pow(2).mean()
                )
            ) / 2

            # Combine data loss and physics loss
            loss = data_loss + lambda_phy * physics_loss
        else:
            loss = data_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
        optimizer.step()

        total_train_loss += loss.item()
        batch_pbar.set_postfix({"batch_loss": f"{loss.item():.6f}"})

    return total_train_loss / len(train_dataloader)


def train_model(
    model_type,
    model,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    optimizer,
    criterion,
    device,
    hyperparameters,
    physics_informed=False,
    lambda_phy=0.1,
    grad_clip_value=1.0,
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
            lambda_phy,
            grad_clip_value,
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
