from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI()

class SatisfactionItem(BaseModel):
    # define data model for the each Item
    YearsAtCompany: float 
    EmployeeSatisfaction: float
    Position: str
    salary: int


with open("rfmodel.pkl", 'rb') as f:
    model = pickle.load(f)
    
"""
we need to find a new model/train a employee churn model (can be simple) 
    - just for deployment, agent and tool purposes

"""

# POST - to enter/send new information(data) to API 
# new data is sent via the SatisfactionItem (Data) model 
@app.get('/')
async def satisfaction_endpoint(item: SatisfactionItem):
    # df = pd.DataFrame([item.dict().values()], columns=item.dict().keys()) # dict() - deprecated
    df = pd.DataFrame([item.model_dump().values()], columns=item.model_dump().keys())
    y_pred = model.predict(df)
    
    return {"Prediction": int(y_pred)}

