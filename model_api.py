from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import joblib

app = FastAPI()

class SatisfactionItem(BaseModel):
    # define data model for the each Item
    YearsAtCompany: float 
    EmployeeSatisfaction: float
    Position: str
    Salary: int


# with open("rfmodel.pkl", 'rb') as f:
#     model = pickle.load(f)
    
try:
    model = joblib.load("ECRFmodel.pkl")  # <- switched from pickle to joblib & new filename
except Exception as e:
    raise RuntimeError(f"Failed to load ECRFmodel.pkl: {e}")

def _normalize_position(pos: str) -> str:
    s = (pos or "").strip().lower()
    if "sales" in s:
        return "Sales"
    if "r&d" in s or "research" in s or "dev" in s:
        return "R&D"
    if "manager" in s:
        return "Manager"
    return "Employee"

def _to_model_df(item: SatisfactionItem) -> pd.DataFrame:
    # Map incoming keys to the exact training feature names
    row = {
        "YearsAtCompany": float(item.YearsAtCompany),
        "EmployeeSatisfaction": float(item.EmployeeSatisfaction),
        "Position": _normalize_position(item.Position),
        "Salary": float(item.Salary), 
    }

    cols = ["YearsAtCompany", "EmployeeSatisfaction", "Position", "Salary"]
    return pd.DataFrame([row], columns=cols)


@app.get('/')
async def test_ping():
    return {"Hello": "Test"}

# POST - to enter/send new information(data) to API 
# new data is sent via the SatisfactionItem (Data) model 
@app.post('/')
async def satisfaction_endpoint(item: SatisfactionItem):
    # df = pd.DataFrame([item.model_dump().values()], columns=item.model_dump().keys())
    try:
        df = _to_model_df(item)
        # If your Pipeline has predict_proba, you can also return probability
        y_pred = model.predict(df)
        result = {"Prediction": int(y_pred[0])}
        # if hasattr(model, "predict_proba"):
        #     prob = float(model.predict_proba(df)[0, 1])
        #     result["Prob"] = round(prob, 3)
        # return result['Prediction']
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")