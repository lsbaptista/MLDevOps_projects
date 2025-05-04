import requests

url = "https://deploying-ml-model-to-render-with-fastapi.onrender.com/predict/"

input_data = {
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

response = requests.post(url, json=input_data)

print("Status code:", response.status_code)
print("Response:", response.json())
