import yaml
import jinja2
from pathlib import Path
import pkg_resources

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
        """Parse the CUIF file and return the configuration."""
        with open(self.input_file, 'r') as f:
            content = f.read()
            # Add YAML document markers
            content = '---\n' + content
            # Parse as YAML
            return yaml.safe_load(content)

    def generate_files(self):
        """Generate all necessary files from the CUIF configuration."""
        config = self.parse_cuif()
        
        # Get the base name from the input file
        base_name = self.input_file.stem

        # Generate .hpp file
        hpp_template = self.env.get_template('header.hpp.jinja2')
        hpp_content = hpp_template.render(**config)
        hpp_file = self.output_dir / f"{base_name}.hpp"
        hpp_file.write_text(hpp_content)
        if self.verbose:
            print(f"Generated {hpp_file}")

        # Generate .cuh file
        cuh_template = self.env.get_template('cuda_header.cuh.jinja2')
        cuh_content = cuh_template.render(**config)
        cuh_file = self.output_dir / f"{base_name}.cuh"
        cuh_file.write_text(cuh_content)
        if self.verbose:
            print(f"Generated {cuh_file}")

        # Generate CMakeLists.txt
        cmake_template = self.env.get_template('CMakeLists.txt.jinja2')
        cmake_content = cmake_template.render(**config)
        cmake_file = self.output_dir / 'CMakeLists.txt'
        cmake_file.write_text(cmake_content)
        if self.verbose:
            print(f"Generated {cmake_file}")

        if self.verbose:
            print("\nGeneration complete!")
            print(f"Generated files in {self.output_dir}:")
            print(f"  - {hpp_file.name}")
            print(f"  - {cuh_file.name}")
            print(f"  - {cmake_file.name}") 