import json
import requests
sample = {"age": 49, "workclass": "Federal-gov", "fnlgt": 125892,
          "education": "Bachelors", "education-num": 13,
          "marital-status": "Married-civ-spouse",
          "occupation": "Exec-managerial",
          "relationship": "Husband", "race": "White", "sex": "Male",
          "capital-gain": 0, "capital-loss": 0, "hours-per-week": 40,
          "native-country": "United-States", "salary": ">50K"}
url = "https://deploying-ml-model.herokuapp.com/inference/"
headers = {"Content-type": "application/json"}
response = requests.post(url, data=json.dumps(sample), headers=headers)
print(response.status_code)
print(response.json())
