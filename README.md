# ODE Transformer for Vehicle Motion Prediction

This project implements a transformer-based model for predicting vehicle motion trajectories using both standard supervised learning and physics-informed approaches.

## Overview

The ODE Transformer model predicts vehicle motion (position, velocity, acceleration) given initial conditions and parameters. Two training approaches are implemented:

1. Standard Training: Uses only supervised learning with data
2. Physics-Informed Training: Combines data-driven learning with physics constraints

The physics-informed version enforces physical relationships between motion variables:

- dx/dt = v (velocity is derivative of position)
- dv/dt = a (acceleration is derivative of velocity)

## Project Structure

odetransformer/
├── architecture/
├── dataset/
├── main.py
├── main_physics.py
├── main_combined.py
└── README.md

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/ode-transformer.git
   cd odetransformer
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Generate data:

   ```bash
   python dataset/generate_data.py
   ```

4. Train models:

   ```bash
   python main.py
   python main_physics.py
   python main_combined.py
   ```

## Training Results

### Standard Training

![Standard Training Loss](results_standard/training_loss.png)

### Physics-Informed Training

![Physics-Informed Training Loss](results_physics/training_loss.png)

### Combined Training

![Combined Training Loss](results_combined/training_loss.png)

## Evaluation

### Standard Model

![Standard Model Prediction](results_standard/predictions.png)

### Physics-Informed Model

![Physics-Informed Model Prediction](results_physics/predictions.png)

### Combined Model

![Combined Model Prediction](results_combined/predictions.png)

## Conclusion

The ODE Transformer model demonstrates improved performance with physics-informed training, capturing physical relationships between motion variables. Standard supervised learning alone shows limited accuracy, while combined training achieves the best results.
