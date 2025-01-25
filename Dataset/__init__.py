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


def create_dataset(dataset_name, params, retina):
    """Create an instance of the specified class."""
    if dataset_name not in _registry:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(_registry.keys())}")
    return _registry[dataset_name](params, retina)


# Dynamically import all modules in this package
package_dir = os.path.dirname(__file__)
for module_info in pkgutil.iter_modules([package_dir]):
    module_name = module_info.name
    importlib.import_module(f"{__name__}.{module_name}")