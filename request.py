import requests

# URL de la API de predicción (asegúrate de que la API esté corriendo)
url = 'http://127.0.0.1:8000/predict-churn'

# Datos de ejemplo de un empleado (esto debe coincidir con las características que espera la API)
employee_data = {
    "Age": 29,
    "BusinessTravel": "Travel_Frequently",
    "Department": "Research & Development",
    "DistanceFromHome": 10,
    "Education": 3,
    "EducationField": "Life Sciences",
    "EnvironmentSatisfaction": 4,
    "Gender": "Male",
    "JobRole": "Research Scientist",
    "JobSatisfaction": 2,
    "MaritalStatus": "Single",
    "MonthlyIncome": 4000,
    "NumCompaniesWorked": 2,
    "OverTime": "Yes",
    "PercentSalaryHike": 13,
    "TotalWorkingYears": 8,
    "TrainingTimesLastYear": 2,
    "WorkLifeBalance": 3,
    "YearsAtCompany": 5,
    "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 2
}

# Realizar la solicitud POST a la API con los datos del empleado
response = requests.post(url, json=employee_data)

# Verificar si la solicitud fue exitosa
if response.status_code == 200:
    prediction = response.json()
    print(f"Predicción: {'Se va' if prediction['prediction'] == 1 else 'Se queda'}")
    print(f"Probabilidades: {prediction['probability']}")
else:
    print(f"Error en la solicitud: {response.status_code}")
    print(response.json())
