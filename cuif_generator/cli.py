#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from .generator import CudaFileGenerator

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

def main():
    parser = argparse.ArgumentParser(description='Generate CUDA files from CUIF specification')
    parser.add_argument('input_file', help='Input CUIF file')
    parser.add_argument('-o', '--output-dir', default='.', help='Output directory')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-d', '--dev', action='store_true', help='Enable development mode with file watching')
    args = parser.parse_args()

    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dev:
        # Development mode: Watch .cuif file and maintain a .cu file
        dev_dir = output_dir / 'dev'
        dev_dir.mkdir(exist_ok=True)
        
        print(f"Development mode enabled. Watching {input_file}...")
        print(f"Temporary .cu file will be maintained at {dev_dir / input_file.stem}.cu")
        print("Press Ctrl+C to stop watching and generate final files.")
        
        event_handler = CuifFileHandler(input_file, dev_dir)
        observer = Observer()
        observer.schedule(event_handler, path=str(input_file.parent), recursive=False)
        observer.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            print("\nGenerating final files...")
        
        observer.join()

    # Generate final files
    generator = CudaFileGenerator(input_file, output_dir, args.verbose)
    generator.generate_files()

if __name__ == '__main__':
    main() 