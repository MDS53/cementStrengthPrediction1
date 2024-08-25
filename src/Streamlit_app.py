import streamlit as st
import numpy as np
import pickle
from Data_Analysis import Visualizations
from Data_Ingestion import Data
from ML_Prediction_Pipeline import Prediction
from Models_info import Models_Info
from utils import get_concrete_reasons

# Set page configuration
st.set_page_config(page_title="Cement Strength Prediction", page_icon="🧱", layout="wide")

# Load data and prediction modules
df=pd.read_excel("Concrete_Data.xls")
X_train,X_test,.y_train,y_test=train_test_split(df.drop(df.columns[-1],axis=1),df[df.columns[-1]])


P = Prediction()

# Custom CSS styling
st.markdown(
    """
    <style>
    /* Main background and text color */
    .main {
        background-color: #000000;
        color: #FAFAFA;
    }
    
    .sidebar .sidebar-content {
        background-color: #225252;
        color: #FAFAFA;
        width: 200px;  /* Adjust this value to change the sidebar width */
    }

    .css-1d391kg {
        width: 200px;  /* Adjust this value to match the sidebar width */
    }

    /* Sidebar text color */
    .css-18e3th9, .css-1v3fvcr, .css-1x7gqtn {
        color: #FAFAFA;
    }

    /* Set the font family */
    body {
        font-family: 'serif';
    }

    /* Header customization */
    .css-18e3th9 h1, .css-18e3th9 h2 {
        color: #FAFAFA;
    }
    </style>
    """, unsafe_allow_html=True
)

# Set the title of the app
st.markdown("<h1 style='text-align: center; font-size: 50px;'>Welcome to Cement Strength Prediction</h1>", unsafe_allow_html=True)

# Initialize session state for navigation buttons if not already set
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

# Function to handle navigation
def navigate_to(page):
    st.session_state.current_page = page

# Create a sidebar with navigation buttons
st.sidebar.title("Navigation")
if st.sidebar.button("Data Analysis"):
    navigate_to("data_analysis")
if st.sidebar.button("Prediction"):
    navigate_to("prediction")
if st.sidebar.button("Models Info"):
    navigate_to("models_info")
if st.sidebar.button("Tips for better strength"):
    navigate_to("tips")

# Display content based on the current page
if st.session_state.current_page == "home":
    st.text("")
    #st.write("This is the home page of the Cement Strength Prediction app. Use the navigation bar to explore different features.")
    st.markdown("<p style='text-align: center; font-size: 20px;'>This is the home page of the Cement Strength Prediction app. Use the navigation bar to explore different features.</p>", unsafe_allow_html=True)
elif st.session_state.current_page == "data_analysis":
    #st.write("You clicked Data Analysis!")
    l = Visualizations(X_train, Y_train, df)
elif st.session_state.current_page == "prediction":
    #st.write("You clicked Prediction")
    P.get_inputs()
elif st.session_state.current_page == "models_info":
    #st.write("You clicked Models Info")
    M = Models_Info()
elif st.session_state.current_page == "tips":
    #st.write("You clicked Tips for better strength")
    g=get_concrete_reasons()
