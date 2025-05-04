# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from starter.ml.data import process_data
import pandas as pd

app = FastAPI()

model = joblib.load('model/model.pkl')


class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        def alias_generator(x):
            return x.replace('_', '-')
        validate_by_name = True

        json_schema_extra = {
            "example": {
                "age": 40,
                "workclass": "Private",
                "fnlgt": 234721,
                "education": "HS-grad",
                "education-num": 9,
                "marital-status": "Married-civ-spouse",
                "occupation": "Tech-support",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }


@app.get("/")
def read_root():
    """
    Handles the root endpoint of the FastAPI application.

    Returns:
        dict: A dictionary containing a welcome message for the ML Inference Service.
    """
    return {"message": "Welcome to the FastAPI ML Inference Service!"}


@app.post("/predict/")
def predict(data: InputData):
    """
    Make a prediction using the input data.

    Args:
        data (InputData): The input data object containing features required for prediction.

    Returns:
        dict: A dictionary containing the prediction label.

    This function processes the input data, encodes categorical features using pre-trained encoders,
    and uses a pre-trained model to make a prediction. The predicted label is then decoded and returned.
    """
    input_data = pd.DataFrame([data.model_dump(by_alias=True)])

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    encoder = joblib.load("model/encoder.pkl")
    lb = joblib.load("model/lb.pkl")
    X, _, _, _ = process_data(input_data, categorical_features=cat_features,
                              label=None, training=False, encoder=encoder, lb=lb)

    prediction = model.predict(X)
    prediction_label = lb.inverse_transform(prediction)[0]

    return {"prediction": prediction_label}
