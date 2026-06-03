"""gRPC server lesson. Mirrors the FastAPI Predict endpoint over HTTP/2.

The proto + generated stubs live at:
    proto/deepcab.proto                 — source schema
    deepCab/grpc/deepcab_pb2.py         — generated message classes (regen via Makefile)
    deepCab/grpc/deepcab_pb2_grpc.py    — generated service stubs

`grpcio-tools` is excluded from the runtime deps (protobuf 5 conflicts with
older mlflow/tf). To (re)generate stubs, run `make grpc_gen` which uses
`uv run --with grpcio-tools ...` to spin up a temporary env.

Once stubs are generated, `python -m deepCab.grpc.server` brings up the server
on :50051. The handler reuses `deepCab.training.predict.predict_one` so REST
and gRPC always return the same result for the same input."""
