import numpy as np
import bentoml
from bentoml.io import NumpyNdarray



wine_clf_runner = bentoml.sklearn.get("wine_clf:latest").to_runner()

svc = bentoml.Service("wine_classifier", runners=[wine_clf_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    return wine_clf_runner.predict.run(input_series)
