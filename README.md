# CUIF Generator

A development tool for generating C++/CUDA/ROS2 interface code from a single, ultra-minimal specification file (`.cuif`).

## Ultra-Minimal Format

Write CUDA code as you normally would, with a minimal YAML header at the top to specify the class and method to expose to ROS2. The rest is just your CUDA/C++ code.

### Example

```yaml
class: DoublerNode
method: double_on_gpu

---
#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel
__global__ void double_array(int* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] *= 2;
  }
}

// Device function (optional)
__device__ int double_value(int x) {
  return x * 2;
}

// The method to be exposed to ROS2
void double_on_gpu(int* device_data, int n) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  double_array<<<blocks, threads>>>(device_data, n);
  cudaDeviceSynchronize();
  std::cout << "Doubled array on GPU!" << std::endl;
}
```

## Usage

### Basic Usage
```bash
cuif-generate examples/doubler.cuif -o output_dir --verbose
```

This will generate:
- `doubler.cu` (your code, ready for ROS2 integration)
- `doubler.hpp` (minimal class declaration)
- `doubler.cuh` (minimal implementation header)
- `CMakeLists.txt` (build config)

### Development Mode
For full IDE support (IntelliSense, syntax highlighting, etc.) while developing:

```bash
cuif-generate examples/doubler.cuif -o output_dir --dev
```

This will:
1. Create a temporary `.cu` file in `output_dir/dev/` with your CUDA code
2. Watch your `.cuif` file for changes
3. Automatically update the `.cu` file when you save changes
4. Generate final files when you press Ctrl+C

Your IDE will recognize the `.cu` file and provide full CUDA development features.

### Code Validation
Validate your CUDA code for common issues and best practices:

```bash
cuif-generate examples/doubler.cuif --validate --verbose
```

The validator checks for:
- Memory management issues
- Kernel launch parameter validation
- Device function usage
- Common CUDA programming errors
- Best practices compliance

### Performance Profiling
Profile your CUDA kernels to analyze performance:

```bash
cuif-generate examples/doubler.cuif --profile --verbose
```

The profiler provides:
- Kernel execution time
- Memory usage statistics
- Grid and block size analysis
- Performance recommendations

## Philosophy
- **Write CUDA code, not boilerplate.**
- **Minimal YAML header** tells the generator what to expose to ROS2.
- **Everything else is just your CUDA/C++ code.**
- The generator wraps your code for ROS2 compatibility.
- **Full IDE support** during development.
- **Built-in validation** ensures code quality.
- **Performance profiling** helps optimize your CUDA kernels.

## Contributing
Pull requests welcome!

## License
MIT 