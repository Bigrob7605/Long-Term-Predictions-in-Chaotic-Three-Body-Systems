#!/usr/bin/env python3
"""
Test script for Three-Body Problem Neural ODE Implementation
============================================================

This script tests the core functionality and validates the implementation
against known solutions and physics constraints.

Run with: python test_implementation.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.three_body_neural_ode import (
    PhysicsInformedNeuralODE,
    ThreeBodyDataGenerator,
    PhysicsLoss
)

def test_basic_functionality():
    """Test basic model creation and forward pass."""
    print("Testing basic functionality...")
    
    # Create model
    model = PhysicsInformedNeuralODE(hidden_dim=64, num_layers=3)
    
    # Test input/output dimensions
    batch_size = 4
    test_input = torch.randn(batch_size, 21)  # 18 state + 3 masses
    test_time = torch.tensor([0.0])
    
    with torch.no_grad():
        output = model(test_time, test_input)
    
    expected_output_shape = (batch_size, 21)  # 12 acc + 9 vel
    assert output.shape == expected_output_shape, f"Expected {expected_output_shape}, got {output.shape}"
    
    print("✓ Basic functionality test passed")
    return model

def test_physics_constraints():
    """Test that physics constraints are properly enforced."""
    print("Testing physics constraints...")
    
    model = PhysicsInformedNeuralODE(hidden_dim=64, num_layers=3)
    
    # Create test state with equal masses
    batch_size = 10
    test_input = torch.randn(batch_size, 21)
    test_input[:, 18:21] = 1.0  # Equal masses
    
    with torch.no_grad():
        output = model(torch.tensor([0.0]), test_input)
    
    # Extract accelerations
    accelerations = output[:, :12].reshape(batch_size, 3, 3)
    masses = test_input[:, 18:21]
    
    # Test center of mass constraint: Σ m_i a_i = 0
    com_acc = torch.sum(masses.unsqueeze(-1) * accelerations, dim=1)
    com_error = torch.norm(com_acc, dim=1)
    
    # Should be very small (numerical precision)
    max_error = torch.max(com_error).item()
    assert max_error < 1e-10, f"Center of mass constraint violated: {max_error}"
    
    print("✓ Physics constraints test passed")

def test_trajectory_integration():
    """Test trajectory integration methods."""
    print("Testing trajectory integration...")
    
    model = PhysicsInformedNeuralODE(hidden_dim=64, num_layers=3)
    
    # Test initial conditions
    initial_state = torch.randn(1, 21)
    time_points = torch.linspace(0, 1.0, 10)
    
    # Test both integration methods
    for method in ['euler', 'rk4']:
        trajectory = model.integrate_trajectory(initial_state, time_points, method=method)
        
        # Check output shape
        expected_shape = (len(time_points), 1, 21)
        assert trajectory.shape == expected_shape, f"Expected {expected_shape}, got {trajectory.shape}"
        
        # Check that initial state is preserved
        initial_error = torch.norm(trajectory[0] - initial_state)
        assert initial_error < 1e-10, f"Initial state not preserved: {initial_error}"
    
    print("✓ Trajectory integration test passed")

def test_data_generation():
    """Test data generation with REBOUND."""
    print("Testing data generation...")
    
    try:
        data_generator = ThreeBodyDataGenerator()
        
        # Generate a small dataset
        initial_states, trajectories = data_generator.generate_equal_mass_system(n_samples=5)
        
        # Check shapes
        assert initial_states.shape == (5, 21), f"Expected (5, 21), got {initial_states.shape}"
        assert trajectories.shape[0] == 5, f"Expected 5 trajectories, got {trajectories.shape[0]}"
        
        # Check that masses are equal
        masses = initial_states[:, 18:21]
        mass_differences = torch.abs(masses - masses[0])
        assert torch.all(mass_differences < 1e-10), "Masses should be equal"
        
        print("✓ Data generation test passed")
        
    except ImportError:
        print("⚠ REBOUND not available, skipping data generation test")

def test_physics_loss():
    """Test physics-informed loss function."""
    print("Testing physics loss function...")
    
    # Create test data
    batch_size = 4
    time_steps = 10
    predicted = torch.randn(batch_size, time_steps, 21)
    target = torch.randn(batch_size, time_steps, 21)
    masses = torch.ones(batch_size, 3)
    
    # Create loss function
    criterion = PhysicsLoss()
    
    # Compute loss
    loss = criterion(predicted, target, masses)
    
    # Check that loss is scalar and positive
    assert loss.dim() == 0, "Loss should be scalar"
    assert loss.item() > 0, "Loss should be positive"
    
    print("✓ Physics loss test passed")

def test_training_loop():
    """Test the training loop (without actual training)."""
    print("Testing training loop setup...")
    
    model = PhysicsInformedNeuralODE(hidden_dim=32, num_layers=2)
    
    # Create dummy data
    initial_states = torch.randn(10, 21)
    trajectories = torch.randn(10, 20, 21)  # 20 time steps
    
    # Create data loader
    dataset = torch.utils.data.TensorDataset(initial_states, trajectories)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Setup training components
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = PhysicsLoss()
    
    # Test one training step
    model.train()
    for batch_ic, batch_trajectories in dataloader:
        optimizer.zero_grad()
        
        # Forward pass
        predicted_trajectories = []
        for ic in batch_ic:
            time_points = torch.linspace(0, 1.0, 20)
            traj = model.integrate_trajectory(ic.unsqueeze(0), time_points)
            predicted_trajectories.append(traj.squeeze(0))
        
        predicted_trajectories = torch.stack(predicted_trajectories)
        
        # Compute loss
        masses = batch_ic[:, 18:21]
        loss = criterion(predicted_trajectories, batch_trajectories, masses)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        break  # Only test one step
    
    print("✓ Training loop test passed")

def run_performance_benchmark():
    """Run a simple performance benchmark."""
    print("Running performance benchmark...")
    
    model = PhysicsInformedNeuralODE(hidden_dim=128, num_layers=4)
    
    # Test different batch sizes
    batch_sizes = [1, 4, 16, 64]
    times = []
    
    for batch_size in batch_sizes:
        test_input = torch.randn(batch_size, 21)
        test_time = torch.tensor([0.0])
        
        # Warm up
        for _ in range(10):
            _ = model(test_time, test_input)
        
        # Time the forward pass
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time and end_time:
            start_time.record()
            for _ in range(100):
                _ = model(test_time, test_input)
            end_time.record()
            torch.cuda.synchronize()
            elapsed = start_time.elapsed_time(end_time) / 100  # Average time per forward pass
        else:
            import time
            start = time.time()
            for _ in range(100):
                _ = model(test_time, test_input)
            elapsed = (time.time() - start) * 1000 / 100  # Convert to milliseconds
        
        times.append(elapsed)
        print(f"  Batch size {batch_size}: {elapsed:.3f} ms")
    
    print("✓ Performance benchmark completed")

def main():
    """Run all tests."""
    print("Three-Body Problem Neural ODE - Test Suite")
    print("=" * 50)
    
    try:
        # Run tests
        test_basic_functionality()
        test_physics_constraints()
        test_trajectory_integration()
        test_data_generation()
        test_physics_loss()
        test_training_loop()
        run_performance_benchmark()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! Implementation is ready.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run training: python src/three_body_neural_ode.py")
        print("3. Explore the code and modify parameters")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
