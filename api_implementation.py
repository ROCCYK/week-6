import json
import requests


url = 'http://127.0.0.1:8000/iris_predict'

input_data_for_model = {
    
    'sepal_length' : 1.2,
    'sepal_width' : 2,
    'petal_length' : 3,
    'petal_width' : 2.2
    }

input_json = json.dumps(input_data_for_model)

response = requests.post(url, data=input_json)
print(response.text)


