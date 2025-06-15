# CUIF Generator for ROS2 CUDA Integration

**DEPRECATED: This is version 1 of the project and is not fully functional. Development has been abandoned in favor of the new and improved version: [robodsl](https://github.com/Zedonkay/robodsl). This repository is kept for historical reference only.**

A tool for generating CUDA-accelerated ROS2 nodes from CUIF (CUDA Interface) files. This tool simplifies the development of GPU-accelerated robotics applications by providing a structured way to define CUDA kernels and device functions.

## Features

- CUDA kernel and device function generation
- ROS2 node integration with clean interface separation
- Automatic generation of:
  - C++ headers (.hpp) with pure virtual interfaces, all constants, and all data structures
  - CUDA headers (.cuh) with device/global function declarations and CUDA-specific structs/constants
  - CUDA source files (.cu) with kernel and device function implementations
- Robust parsing of structs, functions, and globals (constants)
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

Generate CUDA files from one or more CUIF files:
```bash
python -m cuif_generator.cli examples/robot_control.cuif -o output/robot_control
python -m cuif_generator.cli examples/*.cuif -o output/all_controllers
```

### Command Line Options

- `input_files` (positional): One or more CUIF files or glob patterns to process
- `-o`, `--output-dir`: Output directory for generated files (required)
- `-v`, `--verbose`: Enable verbose output showing generated files
- `-d`, `--debug`: Enable debug output showing parsed structs, constants, and methods
- `-h`, `--help`: Show help message and usage examples

#### Example:
```bash
python -m cuif_generator.cli examples/robot_control.cuif -o output/robot_control -v -d
```

### CUIF File Format

CUIF files use a C++-like syntax to define CUDA-accelerated ROS2 nodes. Example:

```cpp
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
__device__ float compute_error(float target, float current);

// Kernels
__global__ void update_state(DeviceState state, float dt);

// Methods
void compute_control(const JointState& current, JointState& target);
```

### Generated Files

The generator creates the following files for each CUIF specification:

1. **`.hpp` file:**
   - Contains all constants and regular data structures
   - Defines pure virtual methods for the ROS2 interface
   - Includes abstract interfaces for device/global functions

2. **`.cuh` file:**
   - Contains CUDA-specific constants and structs
   - Declares device functions and kernels
   - Provides CUDA implementation class

3. **`.cu` file:**
   - Contains the actual CUDA code
   - Implements kernels and device functions
   - Provides ROS2/CUDA integration code

4. **CMakeLists.txt:**
   - Build configuration for the generated files

### Debug and Help

- Use `-d` or `--debug` to print parsed structs, constants, and methods for troubleshooting.
- Use `-h` or `--help` to see all CLI options, usage examples, and CUIF file format documentation.

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
└── README.md
```

### Adding New Controllers

1. Create a new CUIF file in the `examples` directory
2. Define the controller class and method
3. Implement device functions and kernels
4. Add validation and profiling support

## License

This project is licensed under the MIT License - see the LICENSE file for details. 