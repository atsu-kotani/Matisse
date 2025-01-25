import importlib
import pkgutil
import os

# Registry for all classes
_registry = {}

def register_class(name):
    """Decorator to register a class."""
    def decorator(cls):
        _registry[name] = cls
        return cls
    return decorator


def create_logger(logger_name, experiment_name, required_image_resolution):
    """Create an instance of the specified class."""
    if logger_name not in _registry:
        raise ValueError(f"Unknown logger: {logger_name}. Available loggers: {list(_registry.keys())}")
    return _registry[logger_name](experiment_name, required_image_resolution)


# Dynamically import all modules in this package
package_dir = os.path.dirname(__file__)
for module_info in pkgutil.iter_modules([package_dir]):
    module_name = module_info.name
    importlib.import_module(f"{__name__}.{module_name}")