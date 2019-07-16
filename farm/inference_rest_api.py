import logging

import numpy
from flask import Flask, request
from flask_restplus import Api, Resource

from farm.inference import Inferencer

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level="INFO",
    datefmt="%Y-%m-%d %H:%M:%S",
)

INFERENCERS = {}
load_dirs = ["save"]
for idx, model_dir in enumerate(load_dirs):
    INFERENCERS[idx + 1] = Inferencer(model_dir)

app = Flask(__name__)
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


@api.route("/models/<int:model_id>/sequence-classification")
class SequenceClassificationEndpoint(Resource):
    def post(self, model_id):
        model = INFERENCERS.get(model_id, None)
        if not model:
            return "Model not found", 404

        samples = request.get_json().get("input_samples", None)
        if not samples:
            return {}
        raw_data = [sample["texts"] for sample in samples]
        result = model.run_inference(raw_data=raw_data)

        for key, value in result.items():
            if isinstance(value, numpy.floating):
                result[key] = "%.2f" % value
        return {"predictions": result}


if __name__ == "__main__":
    app.run(host="0.0.0.0")
