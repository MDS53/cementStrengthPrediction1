import streamlit as st
import numpy as np
import pickle

class Prediction:
    def __init__(self):
        self.Cement = 0
        self.Blast_Furnace_Slag = 0
        self.Fly_Ash = 0
        self.Water = 0
        self.Superplasticizer = 0
        self.Coarse_Aggregate = 0
        self.Fine_Aggregate = 0
        self.Age = 0
    
    def get_inputs(self):
        with st.form(key='delivery_form_main', clear_on_submit=True):
            self.col1, self.col2 = st.columns([5, 5])
            with self.col1:
                self.Cement = st.number_input("Enter Cement Quantity (Kg in a m^3 mixture)", min_value=50.0, max_value=1000.0)
                self.Fly_Ash = st.number_input("Enter Fly Ash Quantity (Kg in a m^3 mixture)", format="%f", max_value=1000.0)
                self.Superplasticizer = st.number_input("Enter Superplasticizer Quantity (Kg in a m^3 mixture)", format="%f", max_value=1000.0)
                self.Fine_Aggregate = st.number_input("Enter Fine Aggregate Quantity (Kg in a m^3 mixture)", format="%f", max_value=1000.0)
            with self.col2:
                self.Blast_Furnace_Slag = st.number_input("Enter Blast Furnace Slag Quantity (Kg in a m^3 mixture)", format="%f", max_value=1000.0)
                self.Water = st.number_input("Enter Water Quantity (Ltr in a m^3 mixture)", format="%f", min_value=20.0)
                self.Coarse_Aggregate = st.number_input("Enter Coarse Aggregate Quantity (Kg in a m^3 mixture)", format="%f", max_value=1000.0)
                self.Age = st.number_input("Enter Days", min_value=1, max_value=365)
            st.text("Note: Please enter the quantities as per the standard guidelines.And Predicted Values are based on Training Data")
            submit_button = st.form_submit_button(label='Submit')

            if submit_button:
                self.test_data = np.array([
                    self.Cement, self.Blast_Furnace_Slag, self.Fly_Ash, self.Water,
                    self.Superplasticizer, self.Coarse_Aggregate, self.Fine_Aggregate, self.Age
                ]).reshape(1, 8)
                
                self.get_results(self.test_data)
    
    def get_results(self, test_data):
        self.pipe1 = pickle.load(open('pipe1.pkl', 'rb'))
        prediction = self.pipe1.predict(test_data)[0]
        st.write(f"Entered Data: {test_data}")
        st.markdown(f"<h1 style='text-align: center; font-size: 50px;'>Predicted Cement Strength: {np.round(prediction,3)}MPa </h1>", unsafe_allow_html=True)
