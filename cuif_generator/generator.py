import yaml
import jinja2
from pathlib import Path
import pkg_resources
import re

class CudaFileGenerator:
    def __init__(self, input_file, output_dir, verbose=False):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.template_dir = Path(pkg_resources.resource_filename(__name__, 'templates'))
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.template_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )

    def parse_cuif(self):
        """Parse the ultra-minimal CUIF file: YAML header, then verbatim CUDA/C++ code."""
        with open(self.input_file, 'r') as f:
            content = f.read()
        # Split YAML header and code
        if '---' in content:
            header, code = content.split('---', 1)
            config = yaml.safe_load(header)
            config['verbatim_code'] = code.strip()
        else:
            raise ValueError("CUIF file must have a YAML header followed by '---' and code.")
        return config

    def generate_files(self):
        """Generate all necessary files from the CUIF configuration."""
        config = self.parse_cuif()
        base_name = self.input_file.stem

        # Generate .cu file (verbatim code with ROS2/CUDA glue)
        cu_template = self.env.get_template('cuif_minimal.cu.jinja2')
        cu_content = cu_template.render(**config, class_name=config.get('class'), method_name=config.get('method'))
        cu_file = self.output_dir / f"{base_name}.cu"
        cu_file.write_text(cu_content)
        if self.verbose:
            print(f"Generated {cu_file}")

        # Generate .hpp and .cuh files (minimal wrappers)
        hpp_template = self.env.get_template('cuif_minimal.hpp.jinja2')
        hpp_content = hpp_template.render(**config, class_name=config.get('class'), method_name=config.get('method'))
        hpp_file = self.output_dir / f"{base_name}.hpp"
        hpp_file.write_text(hpp_content)
        if self.verbose:
            print(f"Generated {hpp_file}")

        cuh_template = self.env.get_template('cuif_minimal.cuh.jinja2')
        cuh_content = cuh_template.render(**config, class_name=config.get('class'), method_name=config.get('method'))
        cuh_file = self.output_dir / f"{base_name}.cuh"
        cuh_file.write_text(cuh_content)
        if self.verbose:
            print(f"Generated {cuh_file}")

        # Generate CMakeLists.txt
        cmake_template = self.env.get_template('cuif_minimal.CMakeLists.txt.jinja2')
        cmake_content = cmake_template.render(**config, base_name=base_name)
        cmake_file = self.output_dir / 'CMakeLists.txt'
        cmake_file.write_text(cmake_content)
        if self.verbose:
            print(f"Generated {cmake_file}")

        if self.verbose:
            print("\nGeneration complete!")
            print(f"Generated files in {self.output_dir}:")
            print(f"  - {cu_file.name}")
            print(f"  - {hpp_file.name}")
            print(f"  - {cuh_file.name}")
            print(f"  - {cmake_file.name}") 