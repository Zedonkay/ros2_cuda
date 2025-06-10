# CUIF Examples

This directory contains example CUIF files demonstrating various features of the CUIF Generator.

## Example Structure

Each example follows this structure:
```
example_name/
├── controller.cuif    # CUIF specification file
├── controller.hpp     # Generated C++ interface
├── controller.cuh     # Generated CUDA interface
├── controller.cu      # Generated CUDA implementation
└── CMakeLists.txt     # Build configuration
```

## Basic Examples

### robot_control.cuif
A comprehensive example demonstrating GPU-accelerated robot control:
- Constants and data structures for robot state
- Device functions for parallel computation
- CUDA kernels for control updates
- ROS2 interface with pure virtual methods
- Integration of CUDA and ROS2

Usage:
```bash
cuif-generate examples/robot_control.cuif -o output_dir --verbose
```

## Controller Examples

### MPPI Controller (examples/mppi/)
Model Predictive Path Integral control with GPU acceleration:
- Parallel trajectory sampling
- Cost computation using Thrust
- Efficient control updates

### PID Controller (examples/pid/)
Parallel PID control for multiple joints:
- Anti-windup protection
- Parallel joint control
- Efficient integral updates

### LQR Controller (examples/lqr/)
Linear Quadratic Regulator with parallel state feedback:
- Matrix-vector multiplication
- Parallel state feedback
- Efficient gain computation

### MPC Controller (examples/mpc/)
Model Predictive Control with parallel optimization:
- Parallel trajectory optimization
- Efficient gradient computation
- Constraint handling

### RRT Planner (examples/rrt/)
Rapidly-exploring Random Tree planning with GPU acceleration:
- Parallel tree expansion
- Efficient nearest neighbor search
- Collision checking

### Trajectory Follower (examples/trajectory_follower/)
Trajectory following with parallel control:
- Lookahead state computation
- Parallel control computation
- Efficient interpolation

## Testing Features

You can test different features with these examples:

### Validation
```bash
cuif-generate examples/robot_control.cuif --validate --verbose
```

### Profiling
```bash
cuif-generate examples/robot_control.cuif --profile --verbose
```

### Development Mode
```bash
cuif-generate examples/robot_control.cuif -o output_dir --dev
```

## Creating Your Own Examples

When creating your own CUIF files:
1. Start with a minimal YAML header
2. Define constants and data structures
3. Implement device functions and kernels
4. Define ROS2-compatible methods
5. Use the validator to check for issues
6. Profile your kernels for performance
7. Use development mode for IDE support

## Generated Files

Each CUIF file generates three main files:

1. **`.hpp` file:**
   - Contains all constants and data structures
   - Defines pure virtual methods for ROS2 interface
   - Includes abstract interfaces for device/global functions

2. **`.cuh` file:**
   - Contains CUDA-specific constants and structs
   - Implements device functions and kernels
   - Provides CUDA implementations of ROS2 methods

3. **`.cu` file:**
   - Contains the actual CUDA code
   - Implements kernels and device functions
   - Provides ROS2/CUDA integration code 