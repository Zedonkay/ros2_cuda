import yaml
import jinja2
from pathlib import Path
import pkg_resources
import re

class CudaFileGenerator:
    def __init__(self, input_file, output_dir, verbose=False, debug=False):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.debug = debug
        self.template_dir = Path(pkg_resources.resource_filename(__name__, 'templates'))
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.template_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )

    def _debug_print(self, *args, **kwargs):
        """Print debug information only if debug flag is enabled."""
        if self.debug:
            print(*args, **kwargs)

    def parse_cuif(self):
        """Parse the ultra-minimal CUIF file: YAML header, then verbatim CUDA/C++ code."""
        with open(self.input_file, 'r') as f:
            content = f.read()
        # Split YAML header and code
        if '---' in content:
            header, code = content.split('---', 1)
            config = yaml.safe_load(header)
            config['verbatim_code'] = code.strip()
            
            # Parse constants
            config['constants'] = self._parse_constants(code)
            config['cuda_constants'] = self._parse_cuda_constants(code)
            
            # Parse structs
            config['structs'] = self._parse_structs(code)
            config['cuda_structs'] = self._parse_cuda_structs(code)
            
            # Parse device/global functions
            config['device_functions'] = self._parse_device_functions(code)
            config['global_functions'] = self._parse_global_functions(code)
            
            # Parse methods (host/ROS interface)
            config['methods'] = self._parse_methods(code)
        else:
            raise ValueError("CUIF file must have a YAML header followed by '---' and code.")
        return config

    def _parse_constants(self, code):
        """Parse regular constants from the code."""
        constants = []
        const_pattern = r'constexpr\s+(\w+)\s+(\w+)\s*=\s*([^;]+);'
        for match in re.finditer(const_pattern, code):
            type_, name, value = match.groups()
            constants.append({
                'type': type_,
                'name': name,
                'value': value.strip()
            })
        return constants

    def _parse_cuda_constants(self, code):
        """Parse CUDA-specific constants from the code."""
        constants = []
        const_pattern = r'__constant__\s+(\w+)\s+(\w+)\s*=\s*([^;]+);'
        for match in re.finditer(const_pattern, code):
            type_, name, value = match.groups()
            constants.append({
                'type': type_,
                'name': name,
                'value': value.strip()
            })
        return constants

    def _parse_structs(self, code):
        """Parse regular structs from the code, including all members, using brace counting."""
        structs = []
        lines = code.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if 'struct' in line:
                self._debug_print('[DEBUG][structs] line:', repr(line))
            if line.startswith('struct ') and '{' in line:
                # Extract struct name
                name_match = re.match(r'struct\s+(\w+)', line)
                if not name_match:
                    i += 1
                    continue
                name = name_match.group(1)
                struct_lines = []
                brace_count = 1
                i += 1
                while i < len(lines) and brace_count > 0:
                    l = lines[i]
                    struct_lines.append(l)
                    brace_count += l.count('{') - l.count('}')
                    i += 1
                members = []
                for member_line in struct_lines:
                    member_line = member_line.strip()
                    if not member_line or member_line.startswith('//'):
                        continue
                    if member_line in ['public:', 'private:', 'protected:']:
                        continue
                    # Match member: type name; or type name[size];
                    member_match = re.match(r'(\w[\w\s\*&:<>]*)\s+(\w+(\[.*\])?)\s*;', member_line)
                    if member_match:
                        type_, name_ = member_match.groups()[0:2]
                        if '(' in type_ or '(' in name_:
                            continue
                        members.append({'type': type_.strip(), 'name': name_})
                if members:
                    structs.append({'name': name, 'members': members})
            else:
                i += 1
        return structs

    def _parse_cuda_structs(self, code):
        """Parse CUDA-specific structs from the code, including all members, using brace counting."""
        structs = []
        lines = code.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if '__device__ struct' in line:
                self._debug_print('[DEBUG][cuda_structs] line:', repr(line))
            if line.startswith('__device__ struct ') and '{' in line:
                name_match = re.match(r'__device__ struct\s+(\w+)', line)
                if not name_match:
                    i += 1
                    continue
                name = name_match.group(1)
                struct_lines = []
                brace_count = 1
                i += 1
                while i < len(lines) and brace_count > 0:
                    l = lines[i]
                    struct_lines.append(l)
                    brace_count += l.count('{') - l.count('}')
                    i += 1
                members = []
                for member_line in struct_lines:
                    member_line = member_line.strip()
                    if not member_line or member_line.startswith('//'):
                        continue
                    if member_line in ['public:', 'private:', 'protected:']:
                        continue
                    member_match = re.match(r'(\w[\w\s\*&:<>]*)\s+(\w+(\[.*\])?)\s*;', member_line)
                    if member_match:
                        type_, name_ = member_match.groups()[0:2]
                        if '(' in type_ or '(' in name_:
                            continue
                        members.append({'type': type_.strip(), 'name': name_})
                if members:
                    structs.append({'name': name, 'members': members})
            else:
                i += 1
        return structs

    def _parse_device_functions(self, code):
        """Parse device functions from the code."""
        functions = []
        # Match both declarations and definitions
        func_pattern = r'__device__\s+([\w\s\*&:<>]+)\s+(\w+)\s*\(([^)]*)\)(?:\s*;|\s*{)'
        for match in re.finditer(func_pattern, code):
            return_type, name, params_str = match.groups()
            params = []
            if params_str.strip():
                for param in params_str.split(','):
                    param = param.strip()
                    if not param:
                        continue
                    param_match = re.match(r'([\w\s\*&:<>]+)\s+(\w+)', param)
                    if param_match:
                        type_, name_ = param_match.groups()
                        params.append({'type': type_.strip(), 'name': name_})
            functions.append({'return_type': return_type.strip(), 'name': name, 'params': params})
        return functions

    def _parse_global_functions(self, code):
        """Parse global (kernel) functions from the code."""
        functions = []
        # Match both declarations and definitions
        func_pattern = r'__global__\s+([\w\s\*&:<>]+)\s+(\w+)\s*\(([^)]*)\)(?:\s*;|\s*{)'
        for match in re.finditer(func_pattern, code):
            return_type, name, params_str = match.groups()
            params = []
            if params_str.strip():
                for param in params_str.split(','):
                    param = param.strip()
                    if not param:
                        continue
                    param_match = re.match(r'([\w\s\*&:<>]+)\s+(\w+)', param)
                    if param_match:
                        type_, name_ = param_match.groups()
                        params.append({'type': type_.strip(), 'name': name_})
            functions.append({'return_type': return_type.strip(), 'name': name, 'params': params})
        return functions

    def _parse_methods(self, code):
        """Parse class method declarations (not device/global functions, not implementations)."""
        methods = []
        lines = code.splitlines()
        in_class_or_struct = False
        brace_level = 0
        for line in lines:
            stripped = line.strip()
            # Skip comments and preprocessor
            if not stripped or stripped.startswith('//') or stripped.startswith('#'):
                continue
            # Track class/struct scope
            if re.match(r'(class|struct)\s+\w+', stripped):
                in_class_or_struct = True
                continue
            if '{' in stripped:
                brace_level += stripped.count('{')
            if '}' in stripped:
                brace_level -= stripped.count('}')
                if brace_level <= 0:
                    in_class_or_struct = False
                    brace_level = 0
            # Only parse at top level or inside class/struct, not inside function bodies
            if (not in_class_or_struct and brace_level == 0) or (in_class_or_struct and brace_level == 1):
                # Only match lines ending with ';' (declarations)
                if stripped.endswith(';'):
                    # Remove trailing semicolon
                    decl = stripped[:-1].strip()
                    # Match: return_type name(params)
                    method_match = re.match(r'([\w\s\*&:<>~]+)\s+(\w+)\s*\(([^)]*)\)', decl)
                    if method_match:
                        return_type, name, params_str = method_match.groups()
                        # Skip device/global
                        if '__device__' in return_type or '__global__' in return_type:
                            continue
                        # Skip constructors/destructors
                        if name == return_type.split()[-1] or name.startswith('~'):
                            continue
                        # Skip access specifiers
                        if name in ['private', 'protected', 'public']:
                            continue
                        params = []
                        if params_str.strip():
                            for param in params_str.split(','):
                                param = param.strip()
                                if not param:
                                    continue
                                param_match = re.match(r'([\w\s\*&:<>]+)\s+(\w+)', param)
                                if param_match:
                                    type_, name_ = param_match.groups()
                                    params.append({'type': type_.strip(), 'name': name_})
                        methods.append({'return_type': return_type.strip(), 'name': name, 'params': params})
        return methods

    def generate_files(self):
        """Generate all necessary files from the CUIF configuration."""
        config = self.parse_cuif()
        base_name = self.input_file.stem

        # Debug prints
        self._debug_print("[DEBUG] structs:", config.get('structs'))
        self._debug_print("[DEBUG] cuda_structs:", config.get('cuda_structs'))
        self._debug_print("[DEBUG] constants:", config.get('constants'))
        self._debug_print("[DEBUG] cuda_constants:", config.get('cuda_constants'))

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