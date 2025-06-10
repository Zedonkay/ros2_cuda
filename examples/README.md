# CUIF Examples

This directory contains example CUIF files demonstrating various features of the CUIF Generator.

## Basic Examples

### doubler.cuif
A simple example that demonstrates basic CUDA kernel usage:
- Single kernel for array doubling
- Device function for value doubling
- Basic memory management
- Simple kernel launch parameters

Usage:
```bash
cuif-generate examples/doubler.cuif -o output_dir --verbose
```

## Advanced Examples

### thrust_advanced.cuif
Demonstrates advanced CUDA features:
- Thrust library integration
- Multiple kernels
- Complex memory management
- Advanced kernel launch configurations
- Device function usage

Usage:
```bash
cuif-generate examples/thrust_advanced.cuif -o output_dir --verbose
```

## Testing Features

You can test different features with these examples:

### Validation
```bash
cuif-generate examples/doubler.cuif --validate --verbose
```

### Profiling
```bash
cuif-generate examples/thrust_advanced.cuif --profile --verbose
```

### Development Mode
```bash
cuif-generate examples/doubler.cuif -o output_dir --dev
```

## Creating Your Own Examples

When creating your own CUIF files:
1. Start with a minimal YAML header
2. Write your CUDA code as you normally would
3. Use the validator to check for issues
4. Profile your kernels for performance
5. Use development mode for IDE support 