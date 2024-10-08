from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI()

# Cargar el modelo y el scaler
try:
    model = joblib.load('./model/model.pkl')
    scaler = joblib.load('./model/scaler.pkl')
except FileNotFoundError as e:
    raise Exception(f"Archivo no encontrado: {e}")
except Exception as e:
    raise Exception(f"Error al cargar los archivos: {e}")

# Lista de características usadas durante el entrenamiento (incluyendo codificación One-Hot)
original_features = [
    'Age', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 
    'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike', 
    'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
    'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely', 
    'Department_Research & Development', 'Department_Sales', 
    'EducationField_Life Sciences', 'EducationField_Marketing', 
    'EducationField_Medical', 'EducationField_Other', 
    'EducationField_Technical Degree', 'Gender_Male', 
    'JobRole_Human Resources', 'JobRole_Laboratory Technician', 
    'JobRole_Manager', 'JobRole_Manufacturing Director', 
    'JobRole_Research Director', 'JobRole_Research Scientist', 
    'JobRole_Sales Executive', 'JobRole_Sales Representative', 
    'MaritalStatus_Married', 'MaritalStatus_Single', 'OverTime_Yes'
]

def encode_features(features):
    # Codificación one-hot para las características categóricas
    features_encoded = pd.get_dummies(features, drop_first=True)

    # Asegurarse de que las características coincidan con las usadas durante el entrenamiento
    missing_cols = set(original_features) - set(features_encoded.columns)
    for c in missing_cols:
        features_encoded[c] = 0
    
    # Mantener el mismo orden de las columnas que en el entrenamiento
    features_encoded = features_encoded.reindex(columns=original_features, fill_value=0)
    
    return features_encoded

def preprocess_features(data):
    # Crear un DataFrame con las características
    features = pd.DataFrame(data, columns=[col for col in data.keys()])
    
    # Codificar las variables categóricas
    features_encoded = encode_features(features)
    
    # Escalar las características
    features_scaled = scaler.transform(features_encoded)
    
    return features_scaled

class EmployeeData(BaseModel):
    Age: int = Field(..., gt=0, description="Edad del empleado")
    BusinessTravel: str = Field(..., description="Frecuencia de viajes de negocios")
    Department: str = Field(..., description="Departamento del empleado")
    DistanceFromHome: int = Field(..., ge=0, description="Distancia desde el hogar")
    Education: int = Field(..., ge=1, le=5, description="Nivel educativo")
    EducationField: str = Field(..., description="Campo de estudio")
    EnvironmentSatisfaction: int = Field(..., ge=1, le=4, description="Satisfacción con el entorno")
    Gender: str = Field(..., description="Género")
    JobRole: str = Field(..., description="Rol del trabajo")
    JobSatisfaction: int = Field(..., ge=1, le=4, description="Satisfacción con el trabajo")
    MaritalStatus: str = Field(..., description="Estado civil")
    MonthlyIncome: int = Field(..., gt=0, description="Ingreso mensual")
    NumCompaniesWorked: int = Field(..., ge=0, description="Número de empresas trabajadas")
    OverTime: str = Field(..., description="Horas extra trabajadas")
    PercentSalaryHike: int = Field(..., ge=0, description="Porcentaje de aumento salarial")
    TotalWorkingYears: int = Field(..., ge=0, description="Número total de años trabajados")
    TrainingTimesLastYear: int = Field(..., ge=0, description="Cantidad de entrenamientos recibidos en el último año")
    WorkLifeBalance: int = Field(..., ge=1, le=4, description="Balance entre trabajo y vida")
    YearsAtCompany: int = Field(..., ge=0, description="Número de años en la empresa")
    YearsInCurrentRole: int = Field(..., ge=0, description="Número de años en el rol actual")
    YearsSinceLastPromotion: int = Field(..., ge=0, description="Años desde la última promoción")
    YearsWithCurrManager: int = Field(..., ge=0, description="Años con el manager actual")

@app.post('/predict-churn')
def predict_churn(data: EmployeeData):
    try:
        # Convertir los datos en un DataFrame
        features = pd.DataFrame([data.dict()])
        
        # Preprocesar las características
        features_scaled = preprocess_features(features)
        
        # Realizar la predicción
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)
        
        return {
            'prediction': int(prediction[0]),  # 1 para "se va", 0 para "se queda"
            'probability': probability[0].tolist()  # Probabilidades para cada clase
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
