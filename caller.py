import requests

# URL de tu servidor FastAPI
url = 'http://localhost:8000/predict-churn'

# Datos a enviar en la solicitud POST (solo los campos esperados por el modelo)
data = {
    "Age": 29,
    "BusinessTravel": "Travel_Rarely",
    "Department": "Sales",
    "DistanceFromHome": 10,
    "Education": 3,
    "EducationField": "Marketing",
    "EnvironmentSatisfaction": 2,
    "Gender": "Female",
    "JobRole": "Sales Executive",
    "JobSatisfaction": 1,
    "MaritalStatus": "Married",
    "MonthlyIncome": 5000,
    "NumCompaniesWorked": 2,
    "OverTime": "No",
    "PercentSalaryHike": 15,
    "TotalWorkingYears": 5,
    "TrainingTimesLastYear": 2,
    "WorkLifeBalance": 3,
    "YearsAtCompany": 2,
    "YearsInCurrentRole": 1,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 1
}

# Hacer la solicitud POST
response = requests.post(url, json=data)

# Verificar el código de estado de la respuesta
if response.status_code == 200:
    # Imprimir la respuesta en formato JSON
    print(response.json())
else:
    # Imprimir el código de estado y el mensaje de error
    print(f"Error {response.status_code}: {response.text}")
