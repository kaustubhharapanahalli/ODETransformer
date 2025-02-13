import argparse


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
