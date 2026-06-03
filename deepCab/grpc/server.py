"""gRPC server entry. Requires generated stubs (see deepCab/grpc/__init__.py).

Service impl reuses the same `training.predict.predict_one` that powers REST
and the agent's `predict` tool — single source of inference truth across
three transports."""
from __future__ import annotations

import asyncio
from concurrent import futures

import grpc

from deepCab.api.state import STATE
from deepCab.obs.log import get_logger
from deepCab.schemas.data import FeatureRow

log = get_logger(__name__)


def _build_servicer():
    """Import the generated stubs lazily so the module is importable even
    before `make grpc_gen` has been run."""
    try:
        from deepCab.grpc import deepcab_pb2, deepcab_pb2_grpc
    except ImportError as e:
        raise RuntimeError(
            "gRPC stubs not generated yet. Run `make grpc_gen` first."
        ) from e

    class PredictServicer(deepcab_pb2_grpc.PredictServiceServicer):
        def Predict(self, request, context):
            if STATE.model is None:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, "no model loaded")
            row = FeatureRow(
                pickup_datetime=request.row.pickup_datetime,
                pickup_longitude=request.row.pickup_longitude,
                pickup_latitude=request.row.pickup_latitude,
                dropoff_longitude=request.row.dropoff_longitude,
                dropoff_latitude=request.row.dropoff_latitude,
                passenger_count=request.row.passenger_count,
            )
            from deepCab.training.predict import predict_one

            fare = predict_one(STATE.model.estimator, row)
            resp = deepcab_pb2.PredictResponse(
                fare=fare,
                backend_kind=STATE.model.backend_kind,
            )
            if STATE.model.aci is not None:
                import numpy as np
                import pandas as pd

                from deepCab.features.pipeline import preprocess_features

                X = preprocess_features(pd.DataFrame([row.model_dump()])).astype(np.float32)
                _, lo, hi = STATE.model.aci.predict(X)
                resp.interval_lower = float(lo[0])
                resp.interval_upper = float(hi[0])
            return resp

    return PredictServicer, deepcab_pb2_grpc


async def serve(port: int = 50051) -> None:
    servicer_cls, pb_grpc = _build_servicer()
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=4))
    pb_grpc.add_PredictServiceServicer_to_server(servicer_cls(), server)
    server.add_insecure_port(f"[::]:{port}")
    log.info("grpc.serve", port=port)
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
