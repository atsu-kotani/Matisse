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


def create_D_demosaicing(params, device):
    """Create an instance of the specified class."""
    module_type = params['CorticalModel']['cortex_learn_demosaicing']['type']
    if module_type not in _registry:
        raise ValueError(f"Unknown model: {module_type}. Available models: {list(_registry.keys())}")
    return _registry[module_type](params, device)


# Dynamically import all modules in this package
package_dir = os.path.dirname(__file__)
for module_info in pkgutil.iter_modules([package_dir]):
    module_name = module_info.name
    importlib.import_module(f"{__name__}.{module_name}")