from bench.plugin import Plugin, Server
from bench.config import ExperimentSpec, ModelInfo
from bench.plugin import Client
from plugins.triton.triton_client import TritonClient
from server.exporter import ModelExporter
from plugins.triton.triton_server import TritonConfig, TritonServer


class TritonPlugin(Plugin, name="triton"):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def client(self, spec: ExperimentSpec, model: ModelInfo) -> Client:
        triton_config = TritonConfig(model, spec, spec.workspace_dir).create_model_repo(spec.batch_size)
        triton_client = TritonClient(
            "localhost:{}".format(triton_config.http_port),
            model.unique_name(),
            max_paralell_requests=spec.clients,
            metric_tags=spec.metric_tags(),
        )
        return triton_client

    def model(self, spec: ExperimentSpec) -> ModelInfo:
        exporter = ModelExporter(spec)
        return exporter.export()

    def server(self, spec: ExperimentSpec, model: ModelInfo) -> Server:
        triton_config = TritonConfig(model, spec, spec.workspace_dir).create_model_repo(spec.batch_size)
        return TritonServer(triton_config)
