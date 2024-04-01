import numpy as np
import pickle
import streamlit as st

model_file_path = '/Users/athulnambiar/Desktop/PROJECTS/ML-YT/ML1/DISBETES_PREDICTION/trainedmodel.sav'

# Load the model
load_model = pickle.load(open(model_file_path, 'rb'))

def diab_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = load_model.predict(input_data_reshaped)
    return prediction[0]

def main():
    st.title('Diabetes Prediction using ML')
    
    # Input fields for user data
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')
    
    diab_diagnosis = ''
    
    if st.button('Diabetes Test Result'):
        try:
            # Convert input values to float
            Pregnancies = float(Pregnancies)
            Glucose = float(Glucose)
            BloodPressure = float(BloodPressure)
            SkinThickness = float(SkinThickness)
            Insulin = float(Insulin)
            BMI = float(BMI)
            DiabetesPedigreeFunction = float(DiabetesPedigreeFunction)
            Age = float(Age)

            # Perform prediction
            diab_prediction_result = diab_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

            # Interpret prediction
            if diab_prediction_result == 1:
                diab_diagnosis = 'The person is diabetic'
            else:
                diab_diagnosis = 'The person is not diabetic'

            st.success(diab_diagnosis)

        except ValueError:
            st.error('Please enter valid numerical values for all input fields')

if __name__ == '__main__':
    main()
