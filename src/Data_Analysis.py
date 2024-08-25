from Data_Ingestion import Data
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from joypy import joyplot
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from ML_training_pipeline import Pipeline_market
from utils import vif
import inspect
import traceback
from logger import logging
import streamlit as st
import altair as alt

#st.set_page_config(page_title="Cement Analysis", page_icon="🧱",layout="wide")
class PolynomialTransformation:
    def __init__(self,df):
        self.df=df
        self.polynomial_data(self.df)
        
        
    def polynomial_data(self,df):
        try:
            logging.info(f"function : \'{inspect.currentframe().f_code.co_name}\' from  class : \'{self.__class__.__name__} \', Started")
            
            self.df=df
            self.df=self.df[self.df.columns[:-1]]
            self.plf=PolynomialFeatures(degree=3,interaction_only=False)
            self.plf_df=self.plf.fit_transform(self.df)
            self.plf_df=pd.DataFrame(self.plf_df,columns=self.plf.get_feature_names_out())  
            a=[]
            for i in self.plf_df.columns:
                if i.endswith('^3'):
                    a.append(i)
            self.df=self.plf_df[a]
            
            self.sc2=StandardScaler()
            self.df=self.sc2.fit_transform(self.df)
            self.df=pd.DataFrame(self.df, columns=self.sc2.get_feature_names_out())
            logging.info(f"function : \'{inspect.currentframe().f_code.co_name}\' from  class : \'{self.__class__.__name__} \', ended")
            return self.df 
    
        except Exception as e:
            logging.error(f"Exception : {e} at function : \'{inspect.currentframe().f_code.co_name}\' and class : \'{self.__class__.__name__} \', at file {__file__}")
            logging.error(traceback.print_exc())
            
    
     
    def polynomial_splitted(self,X_train,X_test):
        try:
            logging.info(f"function : \'{inspect.currentframe().f_code.co_name}\' from  class : \'{self.__class__.__name__} \', Started")
            self.X_train=X_train
            self.X_test=X_test
            self.plf=PolynomialFeatures(degree=3,interaction_only=False)
            self.plf_X_train=self.plf.fit_transform(self.X_train)
            self.plf_X_test=self.plf.transform(self.X_test)
            self.plf_X_train=pd.DataFrame(self.plf_X_train,columns=self.plf.get_feature_names_out())  
            self.plf_X_test=pd.DataFrame(self.plf_X_test,columns=self.plf.get_feature_names_out())  
            a=[]
            for i in self.plf_df.columns:
                if i.endswith('^3'):
                    a.append(i)
            self.X_train_=self.plf_X_train[a]
            self.X_test_=self.plf_X_test[a]
            self.sc2=StandardScaler()
            self.X_train_=self.sc2.fit_transform(self.X_train_)
            self.X_test_=self.sc2.transform(self.X_test_)
            self.X_train_=pd.DataFrame(self.X_train_, columns=self.sc2.get_feature_names_out())
            self.X_test_=pd.DataFrame(self.X_test_, columns=self.sc2.get_feature_names_out())
            logging.info(f"function : \'{inspect.currentframe().f_code.co_name}\' from  class : \'{self.__class__.__name__} \', ended")
            return self.X_train_,self.X_test_
        
        except Exception as e:
            logging.error(f"Exception : {e} at function : \'{inspect.currentframe().f_code.co_name}\' and class : \'{self.__class__.__name__} \', at file {__file__}")
            logging.error(traceback.print_exc())
            #traceback.print_exc()
    
    def PowerTransformation_data(self,X_train,X_test):
        try:
            logging.info(f"function : \'{inspect.currentframe().f_code.co_name}\' from  class : \'{self.__class__.__name__} \', Started")
            self.X_train_=X_train
            self.X_test_=X_test
            
            self.pp=Pipeline_market()

            self.pipe=self.pp.Power_Transform()
            
            self.X_train=self.pipe.fit_transform(self.X_train_)
            self.X_test=self.pipe.transform(self.X_test_)
            #print("hey")
            self.X_train=pd.DataFrame(self.X_train,columns=self.X_train_.columns)
            self.X_test=pd.DataFrame(self.X_test,columns=self.X_train_.columns)
            logging.info(f"function : \'{inspect.currentframe().f_code.co_name}\' from  class : \'{self.__class__.__name__} \', ended")
            return self.X_train,self.X_test

        except Exception as e:
            logging.error(f"Exception : {e} at function : \'{inspect.currentframe().f_code.co_name}\' and class : \'{self.__class__.__name__} \', at file {__file__}")
            logging.error(traceback.print_exc())




class Visualizations:
    def __init__(self,X_train,Y_train,df):
        self.X_train=X_train
        self.Y_train=Y_train
        self.df=df
        self.get_visuals(self.X_train,self.Y_train,self.df)
        self.multicol_check(self.X_train,self.Y_train,self.df)
    
    def get_visuals(self, X_train, Y_train, df):
        try:
            logging.info(f"Function 'visualize_data' from class '{self.__class__.__name__}', Started")
            
            self.X_train = X_train
            self.Y_train = Y_train
            self.df = df
            
            st.write("Note : Set Your Zoomsize at 75% for better View ")
            #st.title("Data Visualization")
            st.text("")            
            # Slider for selecting number of plots per row
            num_cols = 4
             
            #KDE plots            

            st.markdown("<h1 style='text-align: center; font-size: 40px;'>DENSITY PLOTS</h1>", unsafe_allow_html=True)
            st.write("")
            with st.container():
                cols = st.columns(num_cols)
                
                for idx, col in enumerate(self.df.columns[:-1]):
                    
                    with cols[idx % num_cols]:
                        # Altair KDE Plot
                        kde_plot = alt.Chart(self.df).transform_density(
                            col, as_=[col, 'density']
                        ).mark_area(
                            line={'color': 'darkgreen'},
                            color=alt.Gradient(
                                gradient='linear',
                                stops=[alt.GradientStop(color='white', offset=0),
                                    alt.GradientStop(color='darkgreen', offset=1)],
                                x1=1,
                                x2=1,
                                y1=1,
                                y2=0
                            )
                        ).encode(
                            x=alt.X(col, title=col),
                            y=alt.Y('density:Q')
                        ).properties(width=500, height=400)
                        st.altair_chart(kde_plot)

                    if (idx + 1) % num_cols == 0:
                        st.write("")  # Add a break line for a new row
                     
            st.title("")                        
            st.markdown("<h1 style='text-align: center; font-size: 40px;'>BOX PLOTS</h1>", unsafe_allow_html=True)
            # Box Plots
            #st.subheader("Box Plots")
            st.write("")
            with st.container():
                cols = st.columns(num_cols)
                for idx, col in enumerate(self.X_train.columns):
                    with cols[idx % num_cols]:
                        fig, ax = plt.subplots()
                        sns.boxplot(x=self.X_train[col], ax=ax)
                        st.pyplot(fig)
                    if (idx + 1) % num_cols == 0:
                        st.write("")


            st.title("")  

            #Scatter Plots

            st.markdown("<h1 style='text-align: center; font-size: 40px;'>SCATTER PLOTS</h1>", unsafe_allow_html=True)
            st.write("")
            with st.container():
                cols = st.columns(num_cols)
                for idx, col in enumerate(self.X_train.columns):
                    with cols[idx % num_cols]:
                        # Create a DataFrame for the scatter plot
                        scatter_df = pd.DataFrame({
                            'x': self.X_train[col],
                            'y': self.Y_train
                        })
                        
                        # Altair Scatter Plot
                        scatter_plot = alt.Chart(scatter_df).mark_point().encode(
                            x=alt.X('x', title=col),
                            y=alt.Y('y', title='Target'),
                            color=alt.value('green'),
                            tooltip=['x', 'y']  # Optional: Add tooltips for better interactivity
                        ).properties(
                            width=500,
                            height=400
                        )
                        st.altair_chart(scatter_plot)

                    if (idx + 1) % num_cols == 0:
                        st.write("")  # Add a break line for a new row

            st.title("")  
            # Create a DataFrame for correlation matrix
            correlation_matrix = self.df.corr()
            correlation_df = correlation_matrix.reset_index().melt(id_vars='index')
            correlation_df.columns = ['Variable1', 'Variable2', 'Correlation']
            self.col1, self.col2 = st.columns([5,5]) 
            
            
            
            
            #Heatmap
            with self.col1:
                
                st.subheader("Correlation Heatmap")

                # Heatmap with a vibrant color scheme and annotations
                heatmap = alt.Chart(correlation_df).mark_rect().encode(
                    x=alt.X('Variable1:O', title='',axis=alt.Axis(labelFontSize=13)),
                    y=alt.Y('Variable2:O', title='',axis=alt.Axis(labelFontSize=13)),
                    color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='viridis')),  # Changed to 'plasma' for vibrant colors
                    tooltip=['Variable1', 'Variable2', 'Correlation']
                ).properties(
                    width=1000,
                    height=800
                )

                # Text annotations for correlation values
                text = heatmap.mark_text(baseline='middle',size=15).encode(
                    text=alt.Text('Correlation:Q', format='.2f'),  # Format the text to 2 decimal places
                    color=alt.condition(
                        alt.datum.Correlation > 0.5,
                        alt.value('black'),  # Use black text for higher correlations
                        alt.value('white')   # Use white text for lower correlations
                    )
                )

                # Combine the heatmap and text
                combined_heatmap = heatmap + text

                st.altair_chart(combined_heatmap)
                
                
            with self.col2:    
                # Violin Plot
                st.subheader("Violin Plot for All Features")
                fig, ax = plt.subplots(figsize=(15, 10))
                sns.violinplot(data=self.X_train, palette='Set2', ax=ax)
                ax.set_title('Violin Plot for All Features')
                st.pyplot(fig)
            # Melt the DataFrame for the violin plot
            

            st.title("")  
            st.markdown("<h1 style='text-align: center; font-size: 40px;'>Q-Q PLOTS</h1>", unsafe_allow_html=True)
            st.write("")
            
            with st.container():
                cols = st.columns(num_cols)
                for idx, col in enumerate(self.X_train.columns):
                    with cols[idx % num_cols]:
                        fig, ax = plt.subplots()
                        stats.probplot(self.X_train[col], dist="norm", plot=ax)
                        ax.set_title(f'Q-Q Plot for {col}')
                        st.pyplot(fig)
                    if (idx + 1) % num_cols == 0:
                        st.write("")
            
           
            st.title("")  
            logging.info(f"Function 'visualize_data' from class '{self.__class__.__name__}', Ended")
            
        except Exception as e:
            logging.error(f"Exception : {e} at function : 'visualize_data' and class : '{self.__class__.__name__}', at file {__file__}")
            logging.error(traceback.format_exc())



    def multicol_check(self, X_train, Y_train, df):
        try:
            logging.info(f"Function 'multicol_check' from class '{self.__class__.__name__}', Started")
            
            self.X_train = X_train
            self.Y_train = Y_train
            self.df = df
            
            # Polynomial Transformation and Power Transformation
            self.k = PolynomialTransformation(self.df)
            self.df__ = self.k.polynomial_data(df)
            
           # st.title("Multicollinearity Check")
            st.markdown("<h1 style='text-align: center; font-size: 40px;'>MULTICOLLINEARITY CHECK</h1>", unsafe_allow_html=True)
            st.write("")
            
            # Slider for selecting number of heatmaps per row
            num_cols = 4
            self.pp = Pipeline_market()
            self.pipe = self.pp.Power_Transform()
            self.df_PowerBhai = self.pipe.fit_transform(self.df[self.df.columns[:-1]])
            self.df_PowerBhai = pd.DataFrame(self.df_PowerBhai, columns=self.X_train.columns)
    
            
            
            self.col1, self.col2,self.col3 = st.columns([5,5,5])
            # Create a DataFrame for correlation matrix
            correlation_matrix = self.X_train.corr()
            correlation_df = correlation_matrix.reset_index().melt(id_vars='index')
            correlation_df.columns = ['Variable1', 'Variable2', 'Correlation']
            #self.col1, self.col2 = st.columns([5,5]) 
            
            
            
            #1
            #Heatmap
            with self.col1:
                
                st.subheader("Correlation Heatmap of Original Data")

                # Heatmap with a vibrant color scheme and annotations
                heatmap = alt.Chart(correlation_df).mark_rect().encode(
                    x=alt.X('Variable1:O', title='',axis=alt.Axis(labelFontSize=13)),
                    y=alt.Y('Variable2:O', title='',axis=alt.Axis(labelFontSize=13)),
                    color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='cividis')),  # Changed to 'plasma' for vibrant colors
                    tooltip=['Variable1', 'Variable2', 'Correlation']
                ).properties(
                    width=700,
                    height=800
                )

                # Text annotations for correlation values
                text = heatmap.mark_text(baseline='middle',size=15).encode(
                    text=alt.Text('Correlation:Q', format='.2f'),  # Format the text to 2 decimal places
                    color=alt.condition(
                        alt.datum.Correlation > 0.5,
                        alt.value('black'),  # Use black text for higher correlations
                        alt.value('white')   # Use white text for lower correlations
                    )
                )

                # Combine the heatmap and text
                combined_heatmap = heatmap + text

                st.altair_chart(combined_heatmap)
                
                st.subheader("Variance Inflation Factor (VIF) of Original Data")
                self.h = vif(self.df)
                st.dataframe(self.h,width=10000)
            
            #2
            # Create a DataFrame for correlation matrix
            correlation_matrix = self.df_PowerBhai.corr()
            correlation_df = correlation_matrix.reset_index().melt(id_vars='index')
            correlation_df.columns = ['Variable1', 'Variable2', 'Correlation']
            #self.col1, self.col2 = st.columns([5,5]) 
            
         
            #Heatmap
            with self.col2:
                
                st.subheader("Correlation Heatmap of Power Transformed Data")

                # Heatmap with a vibrant color scheme and annotations
                heatmap = alt.Chart(correlation_df).mark_rect().encode(
                    x=alt.X('Variable1:O', title='',axis=alt.Axis(labelFontSize=13)),
                    y=alt.Y('Variable2:O', title='',axis=alt.Axis(labelFontSize=13)),
                    color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='cividis')),  # Changed to 'plasma' for vibrant colors
                    tooltip=['Variable1', 'Variable2', 'Correlation']
                ).properties(
                    width=700,
                    height=800
                )

                # Text annotations for correlation values
                text = heatmap.mark_text(baseline='middle',size=15).encode(
                    text=alt.Text('Correlation:Q', format='.2f'),  # Format the text to 2 decimal places
                    color=alt.condition(
                        alt.datum.Correlation > 0.5,
                        alt.value('black'),  # Use black text for higher correlations
                        alt.value('white')   # Use white text for lower correlations
                    )
                )

                # Combine the heatmap and text
                combined_heatmap = heatmap + text

                st.altair_chart(combined_heatmap)
                
                st.subheader("Variance Inflation Factor (VIF) of Power Transformed Data")
                self.h = vif(self.df_PowerBhai)
                st.dataframe(self.h,width=10000)
                
                
            
            #3
            # Create a DataFrame for correlation matrix
            correlation_matrix = self.df__.corr()
            correlation_df = correlation_matrix.reset_index().melt(id_vars='index')
            correlation_df.columns = ['Variable1', 'Variable2', 'Correlation']
            #self.col1, self.col2 = st.columns([5,5]) 
            
            
            #Heatmap
            with self.col3:
                
                st.subheader("Correlation Heatmap of Polynomial Transformed Data")

                # Heatmap with a vibrant color scheme and annotations
                heatmap = alt.Chart(correlation_df).mark_rect().encode(
                    x=alt.X('Variable1:O', title='',axis=alt.Axis(labelFontSize=13)),
                    y=alt.Y('Variable2:O', title='',axis=alt.Axis(labelFontSize=13)),
                    color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='cividis')),  # Changed to 'plasma' for vibrant colors
                    tooltip=['Variable1', 'Variable2', 'Correlation']
                ).properties(
                    width=700,
                    height=800
                )

                # Text annotations for correlation values
                text = heatmap.mark_text(baseline='middle',size=15).encode(
                    text=alt.Text('Correlation:Q', format='.2f'),  # Format the text to 2 decimal places
                    color=alt.condition(
                        alt.datum.Correlation > 0.5,
                        alt.value('black'),  # Use black text for higher correlations
                        alt.value('white')   # Use white text for lower correlations
                    )
                )

                # Combine the heatmap and text
                combined_heatmap = heatmap + text

                st.altair_chart(combined_heatmap)
                
                st.subheader("Variance Inflation Factor (VIF) of Polynomial Transformed Data")
                self.h = vif(self.df__)
                st.dataframe(self.h,width=900)
                
                
            
            
            
            
            
            
            
            
            
            logging.info(f"Function 'multicol_check' from class '{self.__class__.__name__}', Ended")
            
        except Exception as e:
            logging.error(f"Exception : {e} at function : 'multicol_check' and class : '{self.__class__.__name__}', at file {__file__}")
            logging.error(traceback.format_exc())

        
    




#k=PolynomialTransformation(df)
#df__=k.polynomial_data(df)
#l=Visualizations(X_train,Y_train,df)




