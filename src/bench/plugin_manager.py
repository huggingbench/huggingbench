import logging
import os
from importlib import import_module
from types import ModuleType
from bench.plugin import Plugin

LOG = logging.getLogger(__name__)

PLUGIN_DIR = "src/plugins"

PLUGINS = []  # pick plugin names from src/plugins folders

for folder in os.listdir(os.path.join(os.getcwd(), PLUGIN_DIR)):
    if os.path.isdir(os.path.join(PLUGIN_DIR, folder)) and folder != "__pycache__":
        PLUGINS.append(folder)


class PluginManager:
    def __init__(self) -> None:
        self.loaded_modules = {}
        self.plugins = {}
        self.load_modules()

    def load_modules(self) -> None:
        for plugin in PLUGINS:
            self.load_module(plugin)

    def load_module(self, name: str) -> ModuleType:
        """Load a plugin module by name"""
        if name not in PLUGINS:
            LOG.error(f"Plugin {name} not found")
            return None
        if name not in self.loaded_modules:
            try:
                module = import_module(f"plugins.{name}.plugin")
                self.loaded_modules[name] = module
                return module
            except Exception as e:
                LOG.error(f"Error loading plugin {name}", exc_info=True)
                return None

    def arg_parsers(self, parsers: dict) -> None:
        for name, plugin_class in Plugin._plugin_classes.items():
            if hasattr(plugin_class, "add_args"):
                plugin_class.add_args(parsers[name])

    def get_plugin(self, name: str, *args, **kwargs) -> Plugin:
        """Picks a plugin by name and returns an instance of it"""
        if name not in self.loaded_modules:
            self.load_module(name)

        plugin_class = Plugin._plugin_classes.get(name)
        if plugin_class is None:
            LOG.error(f"Class for plugin '{name}' not found")
            return None
        if name not in self.plugins:
            self.plugins[name] = plugin_class(args, kwargs)
        return self.plugins[name]
