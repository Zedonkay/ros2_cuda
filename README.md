# CUIF Generator for ROS2 CUDA Integration

A tool for generating CUDA-accelerated ROS2 nodes from CUIF (CUDA Interface) files. This tool simplifies the development of GPU-accelerated robotics applications by providing a structured way to define CUDA kernels and device functions.

## Features

- CUDA kernel and device function generation
- ROS2 node integration with clean interface separation
- Automatic generation of:
  - C++ headers (.hpp) with pure virtual interfaces
  - CUDA headers (.cuh) with device/global function implementations
  - CUDA source files (.cu) with kernel implementations
- Static analysis and validation
- Performance profiling
- Multiple controller implementations:
  - MPPI (Model Predictive Path Integral)
  - PID (Proportional-Integral-Derivative)
  - LQR (Linear Quadratic Regulator)
  - MPC (Model Predictive Control)
  - RRT (Rapidly-exploring Random Tree)
  - Trajectory Follower

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ros2_cuda.git
cd ros2_cuda
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Build the CUDA code:
```bash
mkdir build && cd build
cmake ..
make
```

## Usage

### Basic Usage

Generate CUDA files from a CUIF file:
```bash
python -m cuif_generator.cli --input path/to/file.cuif --output path/to/output/dir
```

### Command Line Options

- `--input`: Path to input CUIF file (required)
- `--output`: Path to output directory (required)
- `--validate`: Enable static analysis and validation
- `--profile`: Enable performance profiling
- `--verbose`: Enable verbose output

### CUIF File Format

CUIF files use a YAML-like syntax to define CUDA-accelerated ROS2 nodes. Here's an example:

```yaml
class: MyNode
method: compute_control
includes:
  - <cuda_runtime.h>
  - <thrust/device_vector.h>
  - <vector>

---
// Constants
constexpr int DOF = 6;
constexpr float PI = 3.14159f;

// Data structures
struct JointState {
    float position;
    float velocity;
};

// CUDA-specific constants
__constant__ float MAX_VELOCITY = 1.0f;

// CUDA-specific structs
__device__ struct DeviceState {
    float* positions;
    float* velocities;
};

// Device functions
__device__ float compute_error(float target, float current) {
    return target - current;
}

// Kernels
__global__ void update_state(DeviceState state, float dt) {
    // Implementation
}

// Methods
void compute_control(const JointState& current, JointState& target) {
    // Implementation
}
```

### Generated Files

The generator creates three main files:

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

### Controller Examples

#### MPPI Controller
The MPPI controller implements Model Predictive Path Integral control with parallel trajectory sampling and optimization. Key features:
- Parallel trajectory sampling
- Cost computation using Thrust
- Efficient control updates

#### PID Controller
The PID controller implements parallel PID control for multiple joints. Key features:
- Anti-windup protection
- Parallel joint control
- Efficient integral updates

#### LQR Controller
The LQR controller implements Linear Quadratic Regulator with parallel state feedback. Key features:
- Matrix-vector multiplication
- Parallel state feedback
- Efficient gain computation

#### MPC Controller
The MPC controller implements Model Predictive Control with parallel gradient descent optimization. Key features:
- Parallel trajectory optimization
- Efficient gradient computation
- Constraint handling

#### RRT Planner
The RRT planner implements Rapidly-exploring Random Tree planning with parallel tree construction. Key features:
- Parallel tree expansion
- Efficient nearest neighbor search
- Collision checking

#### Trajectory Follower
The trajectory follower implements trajectory following with lookahead and parallel control computation. Key features:
- Lookahead state computation
- Parallel control computation
- Efficient interpolation

## Development

### Project Structure

```
ros2_cuda/
├── cuif_generator/
│   ├── cli.py
│   ├── generator.py
│   ├── validator.py
│   ├── profiler.py
│   └── templates/
│       ├── cuif_minimal.hpp.jinja2
│       ├── cuif_minimal.cuh.jinja2
│       └── cuif_minimal.cu.jinja2
├── examples/
│   ├── mppi/
│   ├── pid/
│   ├── lqr/
│   ├── mpc/
│   ├── rrt/
│   ├── trajectory_follower/
│   └── robot_control.cuif
├── tests/
│   ├── test_generator.py
│   ├── test_validator.py
│   └── test_profiler.py
└── README.md
```

### Adding New Controllers

1. Create a new CUIF file in the `examples` directory
2. Define the controller class and method
3. Implement device functions and kernels
4. Add validation and profiling support
5. Update documentation

### Testing

Run tests:
```bash
python -m pytest tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Run tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 