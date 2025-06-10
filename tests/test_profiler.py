import pytest
from pathlib import Path
from cuif_generator.profiler import CuifProfiler, KernelProfile

def test_find_kernels():
    """Test kernel discovery in CUIF file."""
    content = """
class: TestNode
method: test_method

---
__global__ void kernel1(int* data) {
    // Kernel 1
}

__global__ void kernel2(float* data) {
    // Kernel 2
}
"""
    profiler = CuifProfiler(content)
    kernels = profiler.find_kernels()
    assert len(kernels) == 2
    assert 'kernel1' in kernels
    assert 'kernel2' in kernels

def test_kernel_launch_parameters():
    """Test parsing of kernel launch parameters."""
    content = """
class: TestNode
method: test_method

---
__global__ void kernel(int* data) {
    // Kernel
}

void test_method() {
    kernel<<<dim3(2,2,1), dim3(16,16,1)>>>(data);
}
"""
    profiler = CuifProfiler(content)
    profile = profiler.profile_kernel('kernel')
    assert profile is not None
    assert profile.grid_size == (2, 2, 1)
    assert profile.block_size == (16, 16, 1)

def test_profile_report():
    """Test profile report generation."""
    content = """
class: TestNode
method: test_method

---
__global__ void kernel(int* data) {
    // Kernel
}

void test_method() {
    kernel<<<1, 256>>>(data);
}
"""
    profiler = CuifProfiler(content)
    profiler.profile_kernel('kernel')
    report = profiler.get_profile_report()
    assert "CUDA Kernel Profile Report" in report
    assert "Kernel: kernel" in report
    assert "Grid Size: (1, 1, 1)" in report
    assert "Block Size: (256, 1, 1)" in report 