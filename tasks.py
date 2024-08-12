import luigi
import joblib
import pandas as pd
import json

class LoadModel(luigi.Task):
    def output(self):
        return luigi.LocalTarget('model/model.pkl')

    def run(self):
        # La tarea no realiza ninguna acción aquí, simplemente asegura que el archivo esté presente.
        pass

class LoadScaler(luigi.Task):
    def output(self):
        return luigi.LocalTarget('model/scaler.pkl')

    def run(self):
        # La tarea no realiza ninguna acción aquí, simplemente asegura que el archivo esté presente.
        pass

class PredictChurn(luigi.Task):
    data = luigi.DictParameter()

    def requires(self):
        return [LoadModel(), LoadScaler()]

    def output(self):
        return luigi.LocalTarget('prediction_result.json')

    def run(self):
        # Cargar el modelo y el scaler
        model = joblib.load(self.input()[0].path)
        scaler = joblib.load(self.input()[1].path)

        # Preprocesar los datos
        df = pd.DataFrame([self.data])
        features_scaled = preprocess_features(df, scaler)

        # Realizar la predicción
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)

        # Guardar el resultado de la predicción
        result = {
            'prediction': int(prediction[0]),  # 1 para "se va", 0 para "se queda"
            'probability': probability[0].tolist()  # Probabilidades para cada clase
        }
        with open(self.output().path, 'w') as f:
            json.dump(result, f)

def preprocess_features(data, scaler):
    features_encoded = encode_features(data)
    features_scaled = scaler.transform(features_encoded)
    return features_scaled

def encode_features(features):
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
    features_encoded = pd.get_dummies(features, drop_first=True)
    missing_cols = set(original_features) - set(features_encoded.columns)
    for c in missing_cols:
        features_encoded[c] = 0
    features_encoded = features_encoded.reindex(columns=original_features, fill_value=0)
    return features_encoded
