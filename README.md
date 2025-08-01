# Catastrophic Forgetting in Continual Learning Neural Networks

This project investigates the challenge of catastrophic forgetting in neural networks trained sequentially on multiple tasks. Using a PyTorch-based framework, we implemented and compared two continual learning strategies — Elastic Weight Consolidation (EWC) and magnitude-based pruning — across five Permuted-MNIST tasks and 15 synthetic Gaussian mixture datasets.

The framework supports task-wise evaluation, visualizations, and optimized training for CPU environments. Developed as part of a Columbia University research project in collaboration with Mahalakshi Ramakrishnan, Mike Qu and Charan Santhirasegaran.

## Key Features
- Permuted-MNIST and Gaussian mixture benchmarks
- Implementation of EWC (Fisher-based regularization) and pruning
- Matplotlib dashboards for visualizing forgetting curves and performance retention
- Vectorized NumPy + eager-to-static PyTorch conversion for sub-second CPU training
- Fully reproducible experimental pipeline
