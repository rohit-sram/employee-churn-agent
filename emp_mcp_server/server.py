from mcp.server.fastmcp import FastMCP
import json, requests
from typing import List

# Init Server
mcp = FastMCP("EmployeeChurn")

# Create MCP Tool
@mcp.tool
def predict_churn(data: List[dict]) -> str:
    """This tool predicts whether an employee will churn or not, pass through the input as a list of samples.
    Args:
        data: employee attributes which are used for inference. Example payload

        [{
        'YearsAtCompany':10,
        'EmployeeSatisfaction':0.99,
        'Position':'Non-Manager',
        'Salary:5.0
        }]

    Returns:
        str: 1=churn or 0 = no churn"""
    
    PUBLIC_URL = "http://127.0.0.1:8000"
    payload = data[0]
    response = requests.post(
        PUBLIC_URL, headers={"Accept": "application.json", "Content-Type": "application/json"},
        data=json.dumps(payload)
    )
    
    return response.json

if __name__ == '__main__':
    mcp.run(transport="stdio")