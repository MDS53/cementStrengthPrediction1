import pandas as pd
from sklearn.model_selection import train_test_split
import inspect
import traceback
from logger import logging
#from ML_training_pipeline import 


class Data:
        
    def get_split_data(self):
        try:
            logging.info(f"function : \'{inspect.currentframe().f_code.co_name}\' from  class : \'{self.__class__.__name__} \', Started")
            self.df=pd.read_excel("Concrete_Data.xls")
            self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.df.drop(self.df.columns[-1],axis=1),self.df[self.df.columns[-1]])
            logging.info(f"function : \'{inspect.currentframe().f_code.co_name}\' from  class : \'{self.__class__.__name__} \', Ended")
            return self.X_train,self.X_test,self.y_train,self.y_test    
        
        except Exception as e:
            logging.error(f"Exception : {e} at function : \'{inspect.currentframe().f_code.co_name}\' and class : \'{self.__class__.__name__} \', at file {__file__}")
            logging.error(traceback.print_exc())
    
    
    def get_whole_data(self):
        try:
            logging.info(f"function : \'{inspect.currentframe().f_code.co_name}\' from  class : \'{self.__class__.__name__} \', Started")
            self.df=pd.read_excel("C:\\Users\\new\\OneDrive\\Desktop\\Cement Strength prediction\\Data\\Concrete_Data.xls")
            logging.info(f"function : \'{inspect.currentframe().f_code.co_name}\' from  class : \'{self.__class__.__name__} \', Ended")

            return self.df
        except Exception as e:
            logging.error(f"Exception : {e} at function : \'{inspect.currentframe().f_code.co_name}\' and class : \'{self.__class__.__name__} \', at file {__file__}")
            logging.error(traceback.print_exc())

        