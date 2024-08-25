from Data_Ingestion import Data
from Data_Analysis import PolynomialTransformation
from logger import logging
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet,SGDRegressor
from sklearn.ensemble import StackingRegressor,RandomForestRegressor,BaggingRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import inspect
import traceback
import streamlit as st
import altair as alt


class Models_Info:
    def __init__(self):
        self.EBT=pd.read_csv("Errors_before_tunning.csv")
        self.RBT=pd.read_csv("r2_scores_before_tunning.csv")
        
        self.EAT=pd.read_csv("Errors_After_tunning.csv")
        self.RAT=pd.read_csv("r2_scores_After_tunning.csv")
        
        
        self.get_details_visuals(self.RBT,self.EBT,self.EAT,self.RAT)
    
    
    
    def get_details_visuals(self,RBT,EBT,EAT,RAT):
        df_accs=RAT
        df_errors=EAT
        st.text("")
        st.markdown("<h1 style='text-align: center; font-size: 40px;'>R² SCORES </h1>", unsafe_allow_html=True)
        #st.title(" R2scores")
        st.text("")
        st.markdown(
            """
            <div style='font-size:20px; display: flex; align-items: center;'>
                <span style='display:inline-block; width:20px; height:20px; border-radius:50%; background-color:gold; margin-right:10px;'></span>
                <span style='font-weight: bold;'>Train Data</span>
                <span style='display:inline-block; width:20px; height:20px; border-radius:50%; background-color:dodgerblue; margin-right:10px;'></span>
                <span style='font-weight: bold; margin-right:20px;'>Test Data</span>
                
            </div>
            """,
            unsafe_allow_html=True
        )
        st.text("")
        st.text("")
        
        try:
            self.col1, self.col2 = st.columns([5, 5])
            #st.write(" Polynomial vs Power Models")
            background_color = '#f5f5f5'  # Light grey background color
            axis_label_font_size = 14
            #st.title(" R2scores")
            
            # Configure background and axis label font size
            background_color = '#f5f5f5'  # Light grey background color
            axis_label_font_size = 15
            
            
            
            with self.col1:
            # Plot 1: Polynomial Train vs. Test R^2 Scores
            
                # Chart 1: Polynomial Train R^2 Score
                line_chart1 = alt.Chart(df_accs).mark_line().encode(
                    x=alt.X('Algorithmn', title='Algorithm', axis=alt.Axis(labelFontSize=axis_label_font_size)),
                    y=alt.Y('Polynomial_Train_r2score', title='R^2 Score', axis=alt.Axis(labelFontSize=axis_label_font_size)),
                    color=alt.value('gold'),
                    tooltip=['Algorithmn', 'Polynomial_Train_r2score']
                ).properties(
                    title='R² Scores for Polynomial Transformations',
                    width=600,
                    height=600
                )

                points_chart1 = alt.Chart(df_accs).mark_point(
                    filled=True,
                    size=100,
                    color='gold'
                ).encode(
                    x='Algorithmn',
                    y='Polynomial_Train_r2score',
                    tooltip=['Algorithmn', 'Polynomial_Train_r2score']
                )

                # Chart 2: Polynomial Test R^2 Score
                line_chart2 = alt.Chart(df_accs).mark_line().encode(
                    x=alt.X('Algorithmn', title='Algorithm', axis=alt.Axis(labelFontSize=axis_label_font_size)),
                    y=alt.Y('Polynomial_Test_r2score', title='R^2 Score', axis=alt.Axis(labelFontSize=axis_label_font_size)),
                    color=alt.value('dodgerblue'),
                    tooltip=['Algorithmn', 'Polynomial_Test_r2score']
                ).properties(
                    title='Polynomial Test R^2 Score',
                    width=600,
                    height=600
                )

                points_chart2 = alt.Chart(df_accs).mark_point(
                    filled=True,
                    size=100,
                    color='dodgerblue'
                ).encode(
                    x='Algorithmn',
                    y='Polynomial_Test_r2score',
                    tooltip=['Algorithmn', 'Polynomial_Test_r2score']
                )
                

                # Combine the line charts with the point charts
                combined_chart1 = line_chart1 + points_chart1
                combined_chart2 = line_chart2 + points_chart2

                # Display the charts in Streamlit
                st.altair_chart(combined_chart1 + combined_chart2, use_container_width=True)
                # Plot 3: Polynomial Diff (Train-Test)
                chart5 = alt.Chart(df_accs).mark_bar().encode(
                    x=alt.X('Algorithmn', title='Algorithm',axis=alt.Axis(labelFontSize=axis_label_font_size)),
                    y=alt.Y('Polynomial_Diff(Train-Test)', title='Difference',axis=alt.Axis(labelFontSize=axis_label_font_size)),
                    color=alt.value('indigo'),
                    tooltip=['Algorithmn', 'Polynomial_Diff(Train-Test)']
                ).properties(
                    title='R²score Gap(Train - Test) for Polynomial Transformation',
                    width=600,  # Adjust the width as needed
                    height=600  # Adjust the height a
                )

                st.altair_chart(chart5, use_container_width=True)
                
           
            with self.col2:
                # Plot 2: Power Train vs. Test R^2 Scores
                # Chart 3: Power Train R^2 Score
                line_chart3 = alt.Chart(df_accs).mark_line().encode(
                    x=alt.X('Algorithmn', title='Algorithm', axis=alt.Axis(labelFontSize=axis_label_font_size)),
                    y=alt.Y('Power_Train_r2score', title='R^2 Score', axis=alt.Axis(labelFontSize=axis_label_font_size)),
                    color=alt.value('gold'),
                    tooltip=['Algorithmn', 'Power_Train_r2score']
                ).properties(
                    title='R² Scores for Power Transformations',
                    width=600,
                    height=600
                )

                points_chart3 = alt.Chart(df_accs).mark_point(
                    filled=True,
                    size=100,
                    color='gold'
                ).encode(
                    x='Algorithmn',
                    y='Power_Train_r2score',
                    tooltip=['Algorithmn', 'Power_Train_r2score']
                )

                # Chart 4: Power Test R^2 Score
                line_chart4 = alt.Chart(df_accs).mark_line().encode(
                    x=alt.X('Algorithmn', title='Algorithm', axis=alt.Axis(labelFontSize=axis_label_font_size)),
                    y=alt.Y('Power_Test_r2score', title='R^2 Score', axis=alt.Axis(labelFontSize=axis_label_font_size)),
                    color=alt.value('dodgerblue'),
                    tooltip=['Algorithmn', 'Power_Test_r2score']
                ).properties(
                    title='Power Test R^2 Score',
                    width=600,
                    height=600
                )

                points_chart4 = alt.Chart(df_accs).mark_point(
                    filled=True,
                    size=100,
                    color='dodgerblue'
                ).encode(
                    x='Algorithmn',
                    y='Power_Test_r2score',
                    tooltip=['Algorithmn', 'Power_Test_r2score']
                )

                # Combine line charts with points
                combined_chart3 = line_chart3 + points_chart3
                combined_chart4 = line_chart4 + points_chart4

                st.altair_chart(combined_chart3 + combined_chart4, use_container_width=True)

                # Plot 4: Power Diff (Train-Test)
                chart6 = alt.Chart(df_accs).mark_bar().encode(
                    x=alt.X('Algorithmn', title='Algorithm', axis=alt.Axis(labelFontSize=axis_label_font_size)),
                    y=alt.Y('Power_Diff(Train-Test)', title='Difference', axis=alt.Axis(labelFontSize=axis_label_font_size)),
                    color=alt.value('olive   '),
                    tooltip=['Algorithmn', 'Power_Diff(Train-Test)']
                ).properties(
                    title='R²score Gap(Train - Test) for Power Transformation',
                    width=600,
                    height=600
                )

                st.altair_chart(chart6, use_container_width=True)

            st.subheader(" Detailed Information on R2scores of all individual Machine Learning algorithms ")
            st.dataframe(df_accs)


            st.title("")
            
            
            st.markdown("<h1 style='text-align: center; font-size: 50px;'>ERROR METRICS</h1>", unsafe_allow_html=True)
            # Error metrics
            #st.title("Error Metrics")
            self.col1, self.col2 = st.columns([5, 5])
            with self.col1:
                st.subheader('')
                st.subheader("Metrics on Polynomial Transformation")
                st.text('')
                # Plot 1: Polynomial Train vs. Test MAE
                chart7 = alt.Chart(df_errors).mark_line().encode(
                    x=alt.X('Algorithmn', title='Algorithm', axis=alt.Axis(labelFontSize=axis_label_font_size)),
                    y=alt.Y('Polynomial_Train_mae', title='MAE', axis=alt.Axis(labelFontSize=axis_label_font_size)),
                    color=alt.value('gold'),
                    tooltip=['Algorithmn', 'Polynomial_Train_mae']
                ) + alt.Chart(df_errors).mark_point(color='gold',size=100,filled=True).encode(
                    x=alt.X('Algorithmn'),
                    y=alt.Y('Polynomial_Train_mae'),
                    tooltip=['Algorithmn', 'Polynomial_Train_mae']
                ).properties(
                    title='Polynomial Train MAE',
                    width=600,
                    height=600
                )

                chart8 = alt.Chart(df_errors).mark_line().encode(
                    x=alt.X('Algorithmn', title='Algorithm', axis=alt.Axis(labelFontSize=axis_label_font_size)),
                    y=alt.Y('Polynomial_Test_mae', title='MAE', axis=alt.Axis(labelFontSize=axis_label_font_size)),
                    color=alt.value('dodgerblue'),
                    tooltip=['Algorithmn', 'Polynomial_Test_mae']
                ) + alt.Chart(df_errors).mark_point(color='dodgerblue',size=100,filled=True).encode(
                    x=alt.X('Algorithmn'),
                    y=alt.Y('Polynomial_Test_mae'),
                    tooltip=['Algorithmn', 'Polynomial_Test_mae']
                ).properties(
                    title='Polynomial Test MAE',
                    width=600,
                    height=600
                )

                # Combine charts and add a shared legend
                combined_chart = alt.layer(
                    chart7,
                    chart8
                ).resolve_scale(
                    color='independent'
                ).properties(
                    title='Metrics : RMSE (Train vs Test) ',
                    width=600,
                    height=600
                )

                st.altair_chart(combined_chart, use_container_width=True)

                # Plot 2: Polynomial Train vs. Test MSE
                chart9 = alt.Chart(df_errors).mark_line().encode(
                    x=alt.X('Algorithmn', title='Algorithm', axis=alt.Axis(labelFontSize=axis_label_font_size)),
                    y=alt.Y('Polynomial_Train_mse', title='MSE', axis=alt.Axis(labelFontSize=axis_label_font_size)),
                    color=alt.value('gold'),
                    tooltip=['Algorithmn', 'Polynomial_Train_mse']
                ) + alt.Chart(df_errors).mark_point(color='gold',size=100,filled=True).encode(
                    x=alt.X('Algorithmn'),
                    y=alt.Y('Polynomial_Train_mse'),
                    tooltip=['Algorithmn', 'Polynomial_Train_mse']
                ).properties(
                    title='Metrics : RMSE (Train vs Test)',
                    width=600,
                    height=600
                )

                chart10 = alt.Chart(df_errors).mark_line().encode(
                    x=alt.X('Algorithmn', title='Algorithm'),
                    y=alt.Y('Polynomial_Test_mse', title='MSE'),
                    color=alt.value('dodgerblue'),
                    tooltip=['Algorithmn', 'Polynomial_Test_mse']
                ) + alt.Chart(df_errors).mark_point(color='dodgerblue',size=100,filled=True).encode(
                    x=alt.X('Algorithmn'),
                    y=alt.Y('Polynomial_Test_mse'),
                    tooltip=['Algorithmn', 'Polynomial_Test_mse']
                ).properties(
                    title='Polynomial Test MSE',
                    width=600,
                    height=600
                )

                st.altair_chart(chart9 + chart10, use_container_width=True)

                # Plot 3: Polynomial Train vs. Test RMSE
                chart11 = alt.Chart(df_errors).mark_line().encode(
                    x=alt.X('Algorithmn', title='Algorithm', axis=alt.Axis(labelFontSize=axis_label_font_size)),
                    y=alt.Y('Polynomial_Train_rmse', title='RMSE', axis=alt.Axis(labelFontSize=axis_label_font_size)),
                    color=alt.value('gold'),
                    tooltip=['Algorithmn', 'Polynomial_Train_rmse']
                ) + alt.Chart(df_errors).mark_point(color='gold',size=100,filled=True).encode(
                    x=alt.X('Algorithmn'),
                    y=alt.Y('Polynomial_Train_rmse'),
                    tooltip=['Algorithmn', 'Polynomial_Train_rmse']
                ).properties(
                    title='Metrics : RMSE (Train vs Test) ',
                    width=600,
                    height=600
                )

                chart12 = alt.Chart(df_errors).mark_line().encode(
                    x=alt.X('Algorithmn', title='Algorithm', axis=alt.Axis(labelFontSize=axis_label_font_size)),
                    y=alt.Y('Polynomial_Test_rmse', title='RMSE', axis=alt.Axis(labelFontSize=axis_label_font_size)),
                    color=alt.value('dodgerblue'),
                    tooltip=['Algorithmn', 'Polynomial_Test_rmse']
                ) + alt.Chart(df_errors).mark_point(color='dodgerblue',size=100,filled=True).encode(
                    x=alt.X('Algorithmn'),
                    y=alt.Y('Polynomial_Test_rmse'),
                    tooltip=['Algorithmn', 'Polynomial_Test_rmse']
                ).properties(
                    title='Polynomial Test RMSE',
                    width=600,
                    height=600
                )

                st.altair_chart(chart11 + chart12, use_container_width=True)


                
                #st.altair_chart(chart11 + chart12, use_container_width=True)
            with self.col2:
                st.subheader('')
                st.subheader("Metrics on Power Transformation")
                # Plot 4: Power Train vs. Test MAE
                st.text('')
                chart13 = alt.Chart(df_errors).mark_line().encode(
                    x=alt.X('Algorithmn', title='Algorithm', axis=alt.Axis(labelFontSize=axis_label_font_size)),
                    y=alt.Y('Power_Train_mae', title='MAE', axis=alt.Axis(labelFontSize=axis_label_font_size)),
                    color=alt.value('gold'),
                    tooltip=['Algorithmn', 'Power_Train_mae']
                ) + alt.Chart(df_errors).mark_point(color='gold', size=100, filled=True).encode(
                    x=alt.X('Algorithmn'),
                    y=alt.Y('Power_Train_mae'),
                    tooltip=['Algorithmn', 'Power_Train_mae']
                ).properties(
                    title='Metrics : MAE (Train vs Test)',
                    width=600,
                    height=600
                )

                chart14 = alt.Chart(df_errors).mark_line().encode(
                    x=alt.X('Algorithmn', title='Algorithm', axis=alt.Axis(labelFontSize=axis_label_font_size)),
                    y=alt.Y('Power_Test_mae', title='MAE', axis=alt.Axis(labelFontSize=axis_label_font_size)),
                    color=alt.value('dodgerblue'),
                    tooltip=['Algorithmn', 'Power_Test_mae']
                ) + alt.Chart(df_errors).mark_point(color='dodgerblue', size=100, filled=True).encode(
                    x=alt.X('Algorithmn'),
                    y=alt.Y('Power_Test_mae'),
                    tooltip=['Algorithmn', 'Power_Test_mae']
                ).properties(
                    title='Power Test MAE',
                    width=600,
                    height=600
                )

                st.altair_chart(chart13 + chart14, use_container_width=True)

                # Plot 5: Power Train vs. Test MSE
                chart15 = alt.Chart(df_errors).mark_line().encode(
                    x=alt.X('Algorithmn', title='Algorithm', axis=alt.Axis(labelFontSize=axis_label_font_size)),
                    y=alt.Y('Power_Train_mse', title='MSE', axis=alt.Axis(labelFontSize=axis_label_font_size)),
                    color=alt.value('gold'),
                    tooltip=['Algorithmn', 'Power_Train_mse']
                ) + alt.Chart(df_errors).mark_point(color='gold', size=100, filled=True).encode(
                    x=alt.X('Algorithmn'),
                    y=alt.Y('Power_Train_mse'),
                    tooltip=['Algorithmn', 'Power_Train_mse']
                ).properties(
                    title='Metrics : MSE (Train vs Test)',
                    width=600,
                    height=600
                )

                chart16 = alt.Chart(df_errors).mark_line().encode(
                    x=alt.X('Algorithmn', title='Algorithm', axis=alt.Axis(labelFontSize=axis_label_font_size)),
                    y=alt.Y('Power_Test_mse', title='MSE', axis=alt.Axis(labelFontSize=axis_label_font_size)),
                    color=alt.value('dodgerblue'),
                    tooltip=['Algorithmn', 'Power_Test_mse']
                ) + alt.Chart(df_errors).mark_point(color='dodgerblue', size=100, filled=True).encode(
                    x=alt.X('Algorithmn'),
                    y=alt.Y('Power_Test_mse'),
                    tooltip=['Algorithmn', 'Power_Test_mse']
                ).properties(
                    title='Power Test MSE',
                    width=600,
                    height=600
                )

                st.altair_chart(chart15 + chart16, use_container_width=True)

                # Plot 6: Power Train vs. Test RMSE
                chart17 = alt.Chart(df_errors).mark_line().encode(
                    x=alt.X('Algorithmn', title='Algorithm', axis=alt.Axis(labelFontSize=axis_label_font_size)),
                    y=alt.Y('Power_Train_rmse', title='RMSE', axis=alt.Axis(labelFontSize=axis_label_font_size)),
                    color=alt.value('gold'),
                    tooltip=['Algorithmn', 'Power_Train_rmse']
                ) + alt.Chart(df_errors).mark_point(color='gold', size=100, filled=True).encode(
                    x=alt.X('Algorithmn'),
                    y=alt.Y('Power_Train_rmse'),
                    tooltip=['Algorithmn', 'Power_Train_rmse']
                ).properties(
                    title='Metrics : RMSE (Train vs Test) ',
                    width=600,
                    height=600
                )

                chart18 = alt.Chart(df_errors).mark_line().encode(
                    x=alt.X('Algorithmn', title='Algorithm', axis=alt.Axis(labelFontSize=axis_label_font_size)),
                    y=alt.Y('Power_Test_rmse', title='RMSE', axis=alt.Axis(labelFontSize=axis_label_font_size)),
                    color=alt.value('dodgerblue'),
                    tooltip=['Algorithmn', 'Power_Test_rmse']
                ) + alt.Chart(df_errors).mark_point(color='dodgerblue', size=100, filled=True).encode(
                    x=alt.X('Algorithmn'),
                    y=alt.Y('Power_Test_rmse'),
                    tooltip=['Algorithmn', 'Power_Test_rmse']
                ).properties(
                    title='Power Test RMSE',
                    width=600,
                    height=600
                )

                st.altair_chart(chart17 + chart18, use_container_width=True)

            st.subheader(" Detailed Information on Error Metrics of all individual Machine Learning algorithms ")
            st.dataframe(df_errors)
            st.subheader("")
            st.subheader("MAE : Mean Absolute Error, MSE : Mean Squared Error, RMSE : Root Mean Squared Error")
            logging.info(f"function : \'{inspect.currentframe().f_code.co_name}\' from  class : \'{self.__class__.__name__} \', Ended")



        except Exception as e:
            logging.error(f"Exception : {e} at function : \'{inspect.currentframe().f_code.co_name}\' and class : \'{self.__class__.__name__} \', at file {__file__}")
            logging.error(traceback.print_exc())



#MI=Models_Info()
#df_accs,df_errors=MI.get_details()
#print(df_accs)
#print(df_errors)


#df_accs,df_errors=ML_metrics_df(self.Models)    