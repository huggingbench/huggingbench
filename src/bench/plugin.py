from abc import ABC, abstractmethod
import argparse
from typing import Any
from bench.config import ExperimentSpec, ModelInfo


class Client(ABC):
    """Each inference client must inherit from this class."""

    @abstractmethod
    def infer(self, **kwargs) -> Any:
        pass


class Server(ABC):
    """Each inference server must inherit from this class.
    Servers are typically only docker containers that we need to configure and start."""

    @abstractmethod
    def start(self, *args, **kwargs):
        pass

    @abstractmethod
    def stop(self):
        pass


class Plugin:
    """Eeach inference server becomes a plugin. Each plugin must inherit from this class.
    Plugin typically needs to provide server and client classes.
    Check out `triton` plugin for an example."""

    _plugin_classes = {}

    def __init__(self, **kwargs) -> None:
        """Initialize the plugin. Parsed CLI arguments are passed as kwargs."""
        pass

    def __init_subclass__(cls, name: str, **kwargs) -> None:
        """Register each plugin class."""
        super().__init_subclass__(**kwargs)
        Plugin._plugin_classes[name] = cls
        cls._name = name

    def get_name(self) -> str:
        """Return the name of the plugin from passed class attribute"""
        return self._name

    @abstractmethod
    def client(self, spec: ExperimentSpec, model: ModelInfo) -> Client:
        pass

    @abstractmethod
    def model(self, spec: ExperimentSpec) -> ModelInfo:
        pass

    @abstractmethod
    def server(self, spec: ExperimentSpec, model: ModelInfo) -> Server:
        pass

    @abstractmethod
    def add_args(parser: argparse.ArgumentParser):
        pass
