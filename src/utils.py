from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import streamlit as st


def vif(df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]

    return vif_data

def get_concrete_reasons():
    st.subheader("Features")
    reasons_data = {
        "Cement (component 1)": "Correlation value: 0.5 - Increasing the amount of cement in the mixture positively affects the concrete's compressive strength.",
        "Blast Furnace Slag (component 2)": "Correlation value: 0.13 - Adding blast furnace slag has a mild positive effect on the compressive strength.",
        "Fly Ash (component 3)": "Correlation value: -0.11 - Fly ash has a slight negative effect on compressive strength.",
        "Water (component 4)": "Correlation value: -0.29 - Higher water content negatively affects compressive strength.",
        "Superplasticizer (component 5)": "Correlation value: 0.37 - Superplasticizers improve the compressive strength of concrete.",
        "Coarse Aggregate (component 6)": "Correlation value: -0.16 - More coarse aggregate tends to slightly reduce compressive strength.",
        "Fine Aggregate (component 7)": "Correlation value: -0.17 - Increased fine aggregate content can reduce compressive strength.",
        "Age (day)": "Correlation value: 0.33 - Concrete compressive strength improves as the age of the concrete increases.",
        
    }

    for component, description in reasons_data.items():
        with st.expander(component):
            st.write(description)

    st.markdown('<div class="center-button"><a href="/" class="custom-button">Go Back</a></div>', unsafe_allow_html=True)


 
 
 

 
 
 
 
 
 
 
 
 
 
 
 
 
 
        