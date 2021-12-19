from sklearn.model_selection import train_test_split
from cluster import clustering
from find_best_model import best_model
import pickle
from logger import App_Logger
from file_operation import FileOperations

class train:

    """
    This class is used to train select the best model and train the model.
    """

    def __init__(self,data):
        self.data = data
        self.log_writer=App_Logger()
        self.file_object=open("All_logs/train_log.txt",'a+')
        self.file_opn=FileOperations()

    def train_model(self):

        try:
            self.log_writer.log(self.file_object,"Training is Initiated")
            clus = clustering(self.data)                                   #object initialisation
            clus_model=clus.make_clusters()                                #Get Cluster model
            filename='CLUSTER.sav'
            self.file_opn.save_model(clus_model,filename)                  #Save cluster model for prediction phase


            for i in range(len(self.data['cluster_no'].unique())):

                self.log_writer.log(self.file_object,"Training for CLUSTER{}".format(i))
                cluster_data = self.data[self.data['cluster_no'] == i]     #Select datas with cluster i

                x = cluster_data.drop(['OUTPUT', 'cluster_no'], axis=1)
                y = cluster_data['OUTPUT']                                 #Seperate input and output data

                xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.30, random_state=30)
                                                                              #Split data into training and testing
                model = best_model(xtrain, xtest, ytrain, ytest)    #Object initialisation,pass train and test data's
                final_model = model.model_finder()   #Get the best model
                self.log_writer.log(self.file_object, "Final model for implementation is selected")

                filename = 'finalized_model{}.sav'.format(i)
                self.file_opn.save_model(final_model,filename)       #Save the best model
                self.log_writer.log(self.file_object, "The model is saved in file")

        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while training model')
            raise ex