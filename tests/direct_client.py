import json
from pydantic import ValidationError
from fastapi import HTTPException
from api.main import app, health_check, metadata, predict_single, predict_batch, BookInput, BatchInput

class MockResponse:
    def __init__(self, status_code: int, json_data: dict):
        self.status_code = status_code
        self._json_data = json_data

    def json(self):
        return self._json_data

class DirectTestClient:
    """In-process direct test client that bypasses anyio/Starlette hanging on Windows."""
    def __init__(self, app_instance=None):
        self.app = app_instance or app

    def get(self, url: str):
        if url in ["/", ""]:
            return MockResponse(200, self.app.routes[0].endpoint())
        elif url == "/health":
            return MockResponse(200, health_check())
        elif url == "/metadata":
            return MockResponse(200, metadata())
        return MockResponse(404, {"detail": "Not Found"})

    def post(self, url: str, json: dict = None):
        if json is None:
            json = {}

        if url == "/predict":
            try:
                book_in = BookInput(**json)
                res = predict_single(book_in)
                dump_fn = getattr(res, "model_dump", getattr(res, "dict", None))
                return MockResponse(200, dump_fn() if dump_fn else res)
            except HTTPException as he:
                return MockResponse(he.status_code, {"detail": he.detail})
            except ValidationError as ve:
                return MockResponse(422, {"detail": ve.errors()})
            except Exception as ex:
                return MockResponse(500, {"detail": str(ex)})

        elif url == "/predict/batch":
            try:
                batch_in = BatchInput(**json)
                res = predict_batch(batch_in)
                dump_fn = getattr(res, "model_dump", getattr(res, "dict", None))
                return MockResponse(200, dump_fn() if dump_fn else res)
            except HTTPException as he:
                return MockResponse(he.status_code, {"detail": he.detail})
            except ValidationError as ve:
                return MockResponse(422, {"detail": ve.errors()})
            except Exception as ex:
                return MockResponse(500, {"detail": str(ex)})

        return MockResponse(404, {"detail": "Not Found"})
