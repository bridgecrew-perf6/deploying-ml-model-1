import os
import json
import requests

scores = os.system("python starter/model_performance_on_slices.py")

data = scores

r = requests.post(
    "http://127.0.0.1:8000/scores/",
    data=json.dumps(data)
)

print(r.json())
