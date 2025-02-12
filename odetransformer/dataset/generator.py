#! /usr/bin/env python3

"""
This module generates synthetic vehicle motion data for training a Transformer model to solve ODEs.

The problem involves predicting a vehicle's displacement, velocity, and acceleration over time,
given initial conditions (x₀, v₀) and parameters (A, ω, m) where:
- x₀: Initial displacement
- v₀: Initial velocity
- A: Force amplitude
- ω: Angular frequency
- m: Vehicle mass

The underlying ODE system is:
    dx/dt = v
    dv/dt = (A/m)cos(ωt)
    a(t) = (A/m)cos(ωt)
"""

import numpy as np
import torch
from scipy.integrate import solve_ivp
from torch.utils.data import DataLoader, Dataset


def generate_vehicle_motion_data(num_samples=1000, T=10, num_points=100):
    """
    Generates simulated vehicle motion data by solving the ODE system.

    Args:
        num_samples (int): Number of different trajectories to generate
        T (float): Total time duration in seconds
        num_points (int): Number of time points to sample for each trajectory

    Returns:
        tuple: Contains:
            - contexts: Array of shape (num_samples, 5) containing [x₀, v₀, A, ω, m]
            - time_seq: Array of shape (num_points, 1) containing normalized time points
            - states: Array of shape (num_samples, num_points, 3) containing [x, v, a] sequences
    """
    contexts = []  # Each context is [x₀, v₀, A, ω, m]
    states = []  # Each state sequence is a time series of [x, v, a]
    t = np.linspace(0, T, num_points)
    time_seq = (t / T).astype(np.float32).reshape(-1, 1)

    for _ in range(num_samples):
        # Random initial displacement (-10 to 10 meters) and velocity (-5 to 5 m/s)
        x0 = np.random.uniform(-10, 10)
        v0 = np.random.uniform(-5, 5)
        # Random force parameters
        F = np.random.uniform(100, 1000)  # Force amplitude in Newtons
        omega = np.random.uniform(0.1, 1.0)  # Angular frequency in rad/s
        m = np.random.uniform(1000, 4000)  # Mass in kg

        def odefunc(t, state):
            """ODE system defining the vehicle motion"""
            x, v = state
            a = F * np.cos(omega * t) / m
            return [v, a]

        sol = solve_ivp(odefunc, [0, T], [x0, v0], t_eval=t)
        x_sol = sol.y[0]  # displacement
        v_sol = sol.y[1]  # velocity
        a_sol = F * np.cos(omega * t) / m  # acceleration

        # Combine the state variables into a single array: shape (num_points, 3)
        state_seq = np.stack([x_sol, v_sol, a_sol], axis=-1)

        contexts.append([x0, v0, F, omega, m])
        states.append(state_seq)

    return (
        np.array(contexts, dtype=np.float32),
        time_seq.astype(np.float32),
        np.array(states, dtype=np.float32),
    )


class VehicleMotionDataset(Dataset):
    """
    PyTorch Dataset for vehicle motion data with optional normalization.

    Args:
        contexts (np.ndarray): Array of shape (num_samples, 5) containing [x₀, v₀, A, ω, m]
        time_seq (np.ndarray): Array of shape (num_points, 1) containing time points
        states (np.ndarray): Array of shape (num_samples, num_points, 3) containing [x, v, a]
        context_mean (np.ndarray, optional): Mean values for context normalization
        context_std (np.ndarray, optional): Standard deviation values for context normalization
        state_mean (np.ndarray, optional): Mean values for state normalization
        state_std (np.ndarray, optional): Standard deviation values for state normalization
    """

    def __init__(
        self,
        contexts,
        time_seq,
        states,
        context_mean=None,
        context_std=None,
        state_mean=None,
        state_std=None,
    ):
        if context_mean is not None and context_std is not None:
            self.contexts = (contexts - context_mean) / context_std
        else:
            self.contexts = contexts

        if state_mean is not None and state_std is not None:
            self.states = (states - state_mean) / state_std
        else:
            self.states = states

        self.time_seq = time_seq

    def __len__(self):
        """Returns the number of samples in the dataset"""
        return len(self.contexts)

    def __getitem__(self, idx):
        """
        Returns a single sample from the dataset.

        Args:
            idx (int): Index of the sample to return

        Returns:
            tuple: Contains:
                - time sequence tensor of shape (num_points, 1)
                - context tensor of shape (5,)
                - state sequence tensor of shape (num_points, 3)
        """
        context = self.contexts[idx]
        state_seq = self.states[idx]

        return (
            torch.tensor(self.time_seq, dtype=torch.float32),
            torch.tensor(context, dtype=torch.float32),
            torch.tensor(state_seq, dtype=torch.float32),
        )


if __name__ == "__main__":
    # Example: Generate data for a single sample with 1000 time points
    contexts, time_seq, states = generate_vehicle_motion_data(
        num_samples=1, T=10, num_points=1000
    )

    print(contexts)
    print(time_seq)
    print(states)
