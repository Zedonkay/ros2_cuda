# CUIF Generator

A tool for generating CUDA-ROS2 integration files from CUIF (CUDA Interface) specifications.

## Installation

```bash
pip install cuif-generator
```

## Usage

1. Create a `.cuif` file describing your CUDA-ROS2 integration:

```yaml
# example.cuif
namespace: my_package::cuda

class: MyProcessor
  base: rclcpp::Node
  virtual: true
  factory: create
  implementation: MyProcessor_Impl
  doc: |
    My CUDA-accelerated processor node.

  members:
    - type: bool
      name: m_use_gpu
      init: true
      public: true
      doc: Whether to use GPU acceleration

  methods:
    - name: create
      static: true
      return: std::shared_ptr<MyProcessor>
      params: const rclcpp::NodeOptions& options = rclcpp::NodeOptions()
      doc: Creates a new processor instance

# ... rest of the specification
```

2. Generate the files:

```bash
cuif-generate example.cuif
```

This will create:
- `example.hpp`: C++ header file
- `example.cuh`: CUDA header file
- `CMakeLists.txt`: Build configuration

## CUIF Format

The CUIF format is a YAML-like specification that describes your CUDA-ROS2 integration. Key sections:

### Class Definition
```yaml
class: MyClass
  base: rclcpp::Node  # Optional base class
  virtual: true       # Whether the class is virtual
  factory: create     # Factory method name
  implementation: MyClass_Impl  # Implementation class name
  doc: |             # Class documentation
    My class description
```

### Members
```yaml
members:
  - type: bool
    name: m_use_gpu
    init: true
    public: true
    doc: Member documentation
```

### Methods
```yaml
methods:
  - name: my_method
    return: void
    params: int param1, float param2
    virtual: true
    doc: Method documentation
```

### Dependencies
```yaml
includes:
  - <rclcpp/rclcpp.hpp>
  - <cuda_runtime.h>

cmake:
  project: my_project
  cuda_arch: sm_75
  dependencies:
    - CUDA
    - rclcpp
```

## Examples

See the `examples` directory for complete examples:
- `image_processor.cuif`: CUDA-accelerated image processing node
- `particle_filter.cuif`: CUDA-accelerated particle filter
- `neural_network.cuif`: CUDA-accelerated neural network node

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 