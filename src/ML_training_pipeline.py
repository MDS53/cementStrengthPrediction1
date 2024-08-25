from sklearn.preprocessing import PowerTransformer,StandardScaler,PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet,SGDRegressor
from sklearn.ensemble import StackingRegressor,RandomForestRegressor,BaggingRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from Data_Ingestion import Data
import pickle
import inspect
import traceback
from logger import logging


class Pipeline_market:
    def Power_Transform(self):
        try:
            logging.info(f"function : \'{inspect.currentframe().f_code.co_name}\' from  class : \'{self.__class__.__name__} \', Started")
            self.clT=ColumnTransformer(transformers=[
                ('PowerTransformer',PowerTransformer(),[0,1,2,3,4,5,6,7]),
            ],remainder='passthrough'                 )
            
            
            self.clT2=ColumnTransformer(transformers=[
                ('StandardScaler',StandardScaler(),[0,1,2,3,4,5,6,7]),
            ],remainder='passthrough')    



            self.Transformation_pipeline=Pipeline([
                ("transformation",self.clT),
                ("Feature_Scaling",self.clT2),
                
            ])


            pickle.dump(self.Transformation_pipeline,open('Transformation_pipeline.pkl','wb'))
            logging.info(f"function : \'{inspect.currentframe().f_code.co_name}\' from  class : \'{self.__class__.__name__} \', Ended")
            return self.Transformation_pipeline
        
        except Exception as e:
            logging.error(f"Exception : {e} at function : \'{inspect.currentframe().f_code.co_name}\' and class : \'{self.__class__.__name__} \', at file {__file__}")
            logging.error(traceback.print_exc())
    
    
    
    def pipeline(self):
        try:
            logging.info(f"function : \'{inspect.currentframe().f_code.co_name}\' from  class : \'{self.__class__.__name__} \', Started")
            self.data=Data()
            self.X_train,self.X_test,self.y_train,self.y_test=self.data.get_split_data()
            self.clT=ColumnTransformer(transformers=[
                ('PowerTransformer',PowerTransformer(),[0,1,2,3,4,5,6,7]),
            ],remainder='passthrough'                 )
            
            
            self.clT2=ColumnTransformer(transformers=[
                ('StandardScaler',StandardScaler(),[0,1,2,3,4,5,6,7]),
            ],remainder='passthrough')    



            self.Transformation_pipeline=Pipeline([
                ("transformation",self.clT),
                ("Feature_Scaling",self.clT2),
                
            ])
            
            self.Models_with_Hype3={'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=0.01),
            'Lasso': Lasso(alpha=0.0001),
            'ElasticNet': ElasticNet(),
            'SVR': SVR(C= 24, epsilon= 0.432, kernel= "rbf"),
            'RandomForestRegressor': RandomForestRegressor(),
            'ExtraTreesRegressor': ExtraTreesRegressor(),
            'BaggingRegressor': BaggingRegressor(estimator=SVR(C= 24, epsilon= 0.432, kernel= "rbf"), n_estimators=30),
            'StackingRegressor': StackingRegressor(estimators=[('LR', LinearRegression()), ('ls', Lasso()),
                                        ('rd', Ridge(alpha=0.01)), ('svr', SVR(C= 24, epsilon= 0.432, kernel= "rbf")),
                                        ('SGD', SGDRegressor(alpha=0.001,random_state=40, penalty= "l2", max_iter= 1947, learning_rate= "invscaling", eta0= 0.030590900828200946, l1_ratio= 0.35046985203347736))],
                            final_estimator=RandomForestRegressor()),
            'SGD': SGDRegressor(alpha=0.001,random_state=40, penalty= "l2", max_iter= 1947, learning_rate= "invscaling", eta0= 0.030590900828200946, l1_ratio= 0.35046985203347736)}
            
            
            pipe1=Pipeline([
                ("Pipe1",self.clT),
                ("Pipe2",self.clT2),
                ("Pipe3",self.Models_with_Hype3['SVR'])
            ])
            pipe1.fit(self.X_train,self.y_train)
            self.y_pred=pipe1.predict(self.X_test)
            print("Heyyyy")
            pickle.dump(pipe1,open('pipe1.pkl','wb'))
            logging.info(f"function : \'{inspect.currentframe().f_code.co_name}\' from  class : \'{self.__class__.__name__} \', Ended")
            return pipe1


        except Exception as e:
            logging.error(f"Exception : {e} at function : \'{inspect.currentframe().f_code.co_name}\' and class : \'{self.__class__.__name__} \', at file {__file__}")
            logging.error(traceback.print_exc())
 
            
