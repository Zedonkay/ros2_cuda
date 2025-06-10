import yaml

class CuifValidationError(Exception):
    pass

class CuifValidator:
    REQUIRED_FIELDS = ["class", "method", "includes"]
    IMPLEMENTATION_SECTIONS = ["cuda_impl", "implementation", "kernels", "device_functions"]

    @staticmethod
    def validate(cuif_path):
        # Load YAML
        try:
            with open(cuif_path, 'r') as f:
                content = f.read()
                # Add YAML document markers if missing
                if not content.strip().startswith('---'):
                    content = '---\n' + content
                data = yaml.safe_load(content)
        except Exception as e:
            raise CuifValidationError(f"YAML parsing error: {e}")

        # Check required fields
        for field in CuifValidator.REQUIRED_FIELDS:
            if field not in data:
                raise CuifValidationError(f"Missing required field: '{field}'")

        # Check types
        if not isinstance(data["class"], str):
            raise CuifValidationError("'class' must be a string")
        if not isinstance(data["method"], str):
            raise CuifValidationError("'method' must be a string")
        if not isinstance(data["includes"], list):
            raise CuifValidationError("'includes' must be a list of strings")
        for inc in data["includes"]:
            if not isinstance(inc, str):
                raise CuifValidationError("All 'includes' must be strings")

        # Check for at least one implementation section
        if not any(section in data for section in CuifValidator.IMPLEMENTATION_SECTIONS):
            raise CuifValidationError(
                f"At least one implementation section is required: {CuifValidator.IMPLEMENTATION_SECTIONS}")

        # Optionally, add more checks here (e.g., for members, methods, etc.)
        return True 