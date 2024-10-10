# 1. Library imports
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import pandas as pd

# 2. Create the app object
app = FastAPI()
class model_input(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length : float
    petal_width : float
     
pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Welcome to Week#6'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To INFT 41000': f'{name}'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/iris_predict')
def predict_iris(data:model_input):
    data = data.dict()
    sepal_length=data['sepal_length']
    sepal_width=data['sepal_width']
    petal_length=data['petal_length']
    petal_width=data['petal_width']
    print(classifier.predict([[sepal_length,sepal_width,petal_length,petal_width]]))
    prediction = classifier.predict([[sepal_length,sepal_width,petal_length,petal_width]])
    return str(prediction)

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload