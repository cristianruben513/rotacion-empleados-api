# TO-DO se necesita actualizar el modelo entrenado
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib

app = FastAPI()

# Cargar el modelo de rotación de empleados
try:
    model = joblib.load('modelo_rotacion.pkl')
except FileNotFoundError:
    raise Exception("El archivo del modelo 'modelo_rotacion.pkl' no se encontró.")

# Ejemplo de codificación para el campo "department"
department_encoding = {
    "IT": 0,
    "HR": 1,
    "Sales": 2,
    # Agrega otros departamentos según tu dataset
}

class EmployeeData(BaseModel):
    age: int = Field(..., gt=0, description="Edad del empleado")
    salary: float = Field(..., gt=0, description="Salario del empleado")
    years_at_company: int = Field(..., ge=0, description="Años en la empresa")
    department: str = Field(..., description="Departamento del empleado")
    job_satisfaction: float = Field(..., ge=0, le=1, description="Satisfacción laboral (0-1)")
    num_projects: int = Field(..., ge=0, description="Número de proyectos")
    hours_per_week: int = Field(..., ge=0, le=168, description="Horas trabajadas por semana")
    performance_score: float = Field(..., ge=0, le=1, description="Puntuación de rendimiento (0-1)")
    recent_promotions: int = Field(..., ge=0, description="Promociones recientes")
    distance_from_home: float = Field(..., ge=0, description="Distancia desde casa")

@app.post('/predict-churn')
def predict_churn(data: EmployeeData):
    try:
        # Codificar el departamento
        department_encoded = department_encoding.get(data.department)
        if department_encoded is None:
            raise ValueError(f"El departamento '{data.department}' no es válido.")
        
        features = [[
            data.age,
            data.salary,
            data.years_at_company,
            department_encoded,
            data.job_satisfaction,
            data.num_projects,
            data.hours_per_week,
            data.performance_score,
            data.recent_promotions,
            data.distance_from_home
        ]]
        
        # Realizar la predicción
        prediction = model.predict(features)
        probability = model.predict_proba(features)
        
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
