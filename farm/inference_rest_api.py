import json
import logging
from pathlib import Path

import numpy as np
from flask import Flask, request, make_response
from flask_cors import CORS
from flask_restplus import Api, Resource

from farm.infer import Inferencer

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level="INFO",
    datefmt="%Y-%m-%d %H:%M:%S",
)

MODELS_DIRS = ["saved_models", "base_models"]

model_paths = []
for model_dir in MODELS_DIRS:
    path = Path(model_dir)
    if path.is_dir():
        models = [f for f in path.iterdir() if f.is_dir()]
        model_paths.extend(models)

INFERENCERS = {}
for idx, model_dir in enumerate(model_paths):
    INFERENCERS[idx + 1] = Inferencer.load(str(model_dir))

app = Flask(__name__)
CORS(app)
api = Api(app, debug=True, validate=True, version="1.0", title="FARM NLP APIs")
app.config["JSON_SORT_KEYS"] = True
app.config["RESTPLUS_VALIDATE"] = True


@api.route("/models")
class ModelListEndpoint(Resource):
    def get(self):
        resp = []

        for idx, model in INFERENCERS.items():
            _res = {
                "id": idx,
                "name": model.name,
                "prediction_type": model.prediction_type,
                "language": model.language,
            }
            resp.append(_res)

        return resp


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


@api.representation("application/json")
def resp_json(data, code, headers=None):
    resp = make_response(json.dumps(data, cls=NumpyEncoder), code)
    resp.headers.extend(headers or {})
    return resp


@api.route("/models/<int:model_id>/inference")
class InferenceEndpoint(Resource):
    def post(self, model_id):
        model = INFERENCERS.get(model_id, None)
        if not model:
            return "Model not found", 404

        dicts = request.get_json().get("input", None)
        if not dicts:
            return {}
        results = model.inference_from_dicts(dicts=dicts, rest_api_schema=True)
        return results[0]


if __name__ == "__main__":
    app.run(host="0.0.0.0")
