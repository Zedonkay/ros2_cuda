#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from .generator import CudaFileGenerator
from .validator import CuifValidator
# from .profiler import CuifProfiler  # SKIP PROFILER

class CuifFileHandler(FileSystemEventHandler):
    def __init__(self, cuif_path, dev_dir):
        self.cuif_path = cuif_path
        self.dev_dir = dev_dir
        self.last_modified = time.time()
        self.update_dev_file()

    def on_modified(self, event):
        if event.src_path == str(self.cuif_path):
            # Debounce rapid file changes
            current_time = time.time()
            if current_time - self.last_modified > 0.5:  # 500ms debounce
                self.last_modified = current_time
                self.update_dev_file()

    def update_dev_file(self):
        """Update the development .cu file with the contents of the .cuif file."""
        with open(self.cuif_path, 'r') as f:
            content = f.read()
        
        # Split YAML header and code
        if '---' in content:
            _, code = content.split('---', 1)
        else:
            code = content

        # Write to temporary .cu file
        cu_path = self.dev_dir / f"{self.cuif_path.stem}.cu"
        with open(cu_path, 'w') as f:
            f.write(code)

def validate_cuif(cuif_path: Path, verbose: bool = False) -> bool:
    """Validate a CUIF file."""
    with open(cuif_path, 'r') as f:
        content = f.read()
    
    validator = CuifValidator(content)
    results = validator.validate()
    
    if verbose:
        print("\nValidation Results:")
        if results['errors']:
            print("\nErrors:")
            for error in results['errors']:
                print(f"  Line {error.line}: {error.message}")
        if results['warnings']:
            print("\nWarnings:")
            for warning in results['warnings']:
                print(f"  Line {warning.line}: {warning.message}")
    
    return len(results['errors']) == 0

# def profile_cuif(cuif_path: Path, verbose: bool = False) -> None:
#     """Profile a CUIF file."""
#     with open(cuif_path, 'r') as f:
#         content = f.read()
#     
#     profiler = CuifProfiler(content)
#     kernels = profiler.find_kernels()
#     
#     if verbose:
#         print(f"\nFound {len(kernels)} kernels:")
#         for kernel in kernels:
#             print(f"  - {kernel}")
#     
#     for kernel in kernels:
#         profiler.profile_kernel(kernel)
#     
#     print(profiler.get_profile_report())

def main():
    parser = argparse.ArgumentParser(
        description='CUIF Generator - Generate CUDA/ROS2 interface files from CUIF specifications',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Generate files for a single controller
  python -m cuif_generator.cli examples/robot_control.cuif -o output/robot_control

  # Generate files with verbose output and debug information
  python -m cuif_generator.cli examples/robot_control.cuif -o output/robot_control -v -d

  # Generate files for all controllers in a directory
  python -m cuif_generator.cli examples/*.cuif -o output

File Structure:
  The generator creates the following files for each CUIF specification:
  - <name>.hpp    : C++ header with class declarations and regular structs
  - <name>.cuh    : CUDA header with device functions and CUDA-specific structs
  - <name>.cu     : CUDA implementation file
  - CMakeLists.txt: Build configuration

CUIF File Format:
  A CUIF file should contain:
  - Regular structs: struct Name { ... };
  - CUDA structs: __device__ struct Name { ... };
  - Constants: constexpr type name = value;
  - CUDA constants: __constant__ type name = value;
  - Class methods and device functions
'''
    )
    
    parser.add_argument('input_files', nargs='+', type=str,
                      help='CUIF specification file(s) to process. Can be a single file or glob pattern.')
    
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                      help='Output directory for generated files')
    
    parser.add_argument('-v', '--verbose', action='store_true',
                      help='Enable verbose output showing generated files')
    
    parser.add_argument('-d', '--debug', action='store_true',
                      help='Enable debug output showing parsed structs, constants, and methods')
    
    args = parser.parse_args()
    
    # Convert input files to Path objects
    input_files = []
    for pattern in args.input_files:
        input_files.extend(Path().glob(pattern))
    
    if not input_files:
        print(f"Error: No CUIF files found matching patterns: {args.input_files}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each input file
    for input_file in input_files:
        if not input_file.exists():
            print(f"Error: Input file not found: {input_file}")
            continue
            
        if not input_file.suffix == '.cuif':
            print(f"Warning: Skipping non-CUIF file: {input_file}")
            continue
            
        # Create output subdirectory for this file
        file_output_dir = output_dir / input_file.stem
        file_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate files
        generator = CudaFileGenerator(input_file, file_output_dir, verbose=args.verbose, debug=args.debug)
        generator.generate_files()

if __name__ == '__main__':
    main() 