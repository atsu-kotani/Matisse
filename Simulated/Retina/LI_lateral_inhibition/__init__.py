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


def create_lateral_inhibition_module(params, device):
    """Create an instance of the specified class."""
    module_type = params['RetinaModel']['retina_lateral_inhibition']['type']
    if module_type not in _registry:
        raise ValueError(f"Unknown module type: {module_type}. Available modules: {list(_registry.keys())}")
    return _registry[module_type](params, device)


# Dynamically import all modules in this package
package_dir = os.path.dirname(__file__)
for module_info in pkgutil.iter_modules([package_dir]):
    module_name = module_info.name
    importlib.import_module(f"{__name__}.{module_name}")