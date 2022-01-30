import json
import requests
url = "https://deploying-ml-model.herokuapp.com/inference/sex"
response = requests.get(url)
print(response.status_code)
print(json.dumps(response.json(), indent=4))
