### Simple load generation for Triton client
### Primarily used for debugging and testing because Locust does not play well with debuger
import argparse
from client.runner import Runner, RunnerConfig, get_dataset
from client.triton_client import TritonClient

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("-s", "--server", type=str, default="localhost:8000", help="Triton server address")
    parser.add_argument("-c", "--concurrency", type=int, default=1, help="Number of concurrent requests")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("-a", "--async_req", type=bool, default=False, help="Async requests")
    FLAGS = parser.parse_args()

    dataset = get_dataset(FLAGS.model)
    cfg = RunnerConfig(FLAGS.model, FLAGS.batch_size, FLAGS.async_req)
    client = TritonClient(FLAGS.server, FLAGS.model, FLAGS.concurrency)
    runner = Runner(cfg, client, dataset)
    runner.run()
