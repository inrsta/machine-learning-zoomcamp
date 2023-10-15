import requests

data = [{"job": "retired", "duration": 445, "poutcome": "success"}]

response = requests.post("http://127.0.0.1:8080/predict", json=data)
print(response.json())