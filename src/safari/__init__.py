import os
import pkgutil
import importlib
import inspect
import sys

package_path = os.path.dirname(__file__)

for _, module_name, is_pkg in pkgutil.iter_modules([package_path]):
    if not is_pkg and module_name != '__init__':
        module = importlib.import_module(f".{module_name}", package=__name__)

        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) or inspect.isclass(obj):
                # Expose both functions and classes
                setattr(sys.modules[__name__], name, obj)

# Dynamically define __all__ for exposed names
__all__ = [name for name in dir() if not name.startswith('_')]
