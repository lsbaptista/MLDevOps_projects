import requests
import json
import os


URL = "http://127.0.0.1:8000/"


response1 = requests.post(
    URL + "prediction", json={"filepath": "ingesteddata/finaldata.csv"})
response2 = requests.get(URL + "scoring")
response3 = requests.get(URL + "summarystats")
response4 = requests.get(URL + "diagnostics")


responses = {
    "prediction": response1.json() if response1.status_code == 200 else {"error": response1.text},
    "scoring": response2.json() if response2.status_code == 200 else {"error": response2.text},
    "summarystats": response3.json() if response3.status_code == 200 else {"error": response3.text},
    "diagnostics": response4.json() if response4.status_code == 200 else {"error": response4.text}
}


script_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(script_dir, 'config.json'), 'r') as f:
    config = json.load(f)

output_file_path = os.path.join(
    script_dir, config['output_model_path'], 'apireturns.txt')

with open(output_file_path, 'w') as f:
    json.dump(responses, f, indent=4)

print(f"API responses have been written to {output_file_path}")
