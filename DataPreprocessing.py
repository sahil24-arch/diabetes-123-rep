from sklearn.preprocessing import StandardScaler
import pandas as pd
from logger import App_Logger
import pickle

class scale_data:

    """
    This class is used to perform standard scaler operation on given dataset.

    """

    def __init__(self,path):
        self.path=path
        self.log_writer=App_Logger()
        self.file_object=open('All_logs/train_log.txt','a+')
        self.stand=StandardScaler()

    def train_scale(self):

        """
        This method scales the training dataset
        """
        try:
            df=pd.read_csv(self.path)
            self.log_writer.log(self.file_object, 'Scaling dataset')

            X=df.drop('Diabetes_binary',axis=1)
            Y=df['Diabetes_binary']                   #Seperate input and output dataset

            df_scaled=self.stand.fit_transform(X)     #Scale the input dataset.


            df_new=pd.DataFrame(df_scaled,columns=list(X.columns))    #Convert numpy array to Dataframe
            df_new['OUTPUT']=Y                                        #Add output column to scaled dataset

            return df_new                   #Return dataset
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured during scaling training data')
            raise ex

    def pred_scale(self):

        """
               This method scales the prediction dataset
               """

        try:
            df = pd.read_csv(self.path)                       #Convert prediction data csv file to Dataframe
            self.log_writer.log(self.file_object, 'Scaling dataset')

            X = df.drop('Patient', axis=1)
            col = df['Patient']                    #Seperate the patient column from input data

            df_scaled = self.stand.fit_transform(X)      #Scale the input dataset

            df_new = pd.DataFrame(df_scaled, columns=list(X.columns))     #Convert numpy array to Dataframe

            df_new['Patient'] = col    #Add patient column to the dcaled data

            return df_new              #return data

        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured during scaling prediction data')
            raise ex