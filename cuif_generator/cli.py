#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from .generator import CudaFileGenerator

def main():
    parser = argparse.ArgumentParser(
        description='Generate CUDA-ROS2 integration files from CUIF specification'
    )
    parser.add_argument(
        'input_file',
        help='Input .cuif file'
    )
    parser.add_argument(
        '--output-dir',
        '-o',
        help='Output directory (default: same as input file)',
        type=Path
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose output'
    )
    args = parser.parse_args()

    input_file = Path(args.input_file)
    if not input_file.exists():
        parser.error(f"Input file {input_file} does not exist")

    output_dir = args.output_dir or input_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = CudaFileGenerator(input_file, output_dir, verbose=args.verbose)
    generator.generate_files()

if __name__ == '__main__':
    main() 