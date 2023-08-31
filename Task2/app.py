# Author: CHEN Ee Heng
# Date: 30.08.2023

from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse

from .inference import Classifier
from .schema import RequestItem

clf = Classifier()
app = FastAPI(title="Classification Inferencer")


@app.get("/")
async def welcome():
    return {"msg":
            "This is a Classification Inferencer API. Go to docs to start !!!"}


@app.exception_handler(AssertionError)
async def assert_error_exception_handler(request: Request, exc: AssertionError):
    return JSONResponse(
        status_code=400,
        content={"assert error message": str(exc)},
    )


@app.post("/predict/", status_code=200)
async def predict_category(request: RequestItem):
    prediction = clf.predict(clf.preprocess(request.input_data))
    if request.convert_label_id_to_name:
        prediction = clf.postprocess(prediction)
    return {"status_code": 200,
            "prediction": prediction,
            "label_id": clf.label_id if request.return_label_ids else None}
