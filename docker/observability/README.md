# Observability stack

For tracking benchmarks statistics and visualizing results we use Prometheus and Grafana.
While benchmarks are running we are collecting both server/Triton and client metrics.
Make sure to spin up the stack before running the benchmarks.

`docker-compose up` or `docker compose up`

On Linux run (must run from this folder): `./start-docker-compose.sh`


For Grafana go to: `http://localhost:3000` and login: `admin/foobar`
For Prometheus go to: `http://localhost:9090`

Grafana is preloaded with HuggingBench Main Dashboard.
