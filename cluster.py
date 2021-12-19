from sklearn.cluster import KMeans
from kneed import KneeLocator
from logger import App_Logger
import pickle

class clustering:

    """
    This class is used to create cluster of given dataset
    """

    def __init__(self, data):
        self.data = data
        self.cdata = self.data.drop('OUTPUT', axis=1)
        self.log_writer=App_Logger()
        self.file_object=open('All_logs/train_log.txt','a+')
        print('clustering started')

    def find_clusters(self):

        """
        This method is used to find how many clusters can be created.
        """

        try:
            self.kmean = KMeans()             #Object initialisation
            inertia = []                      #empty list to save inertia(wcss)

            for i in range(1, 11):
                self.kmean = KMeans(n_clusters=i, random_state=30)
                self.kmean.fit(self.cdata)               #create i clusters
                inertia.append(self.kmean.inertia_)      #append inertia value to list

            self.kn = KneeLocator(range(1, 11), inertia, curve='convex', direction='decreasing')    #To get the value of no. of clusters
            self.log_writer.log(self.file_object, "We get number of clusters to be formed,{}".format(self.kn.knee))

            return self.kn.knee   #return number of cluster

        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured during finding cluster numbers')
            raise  ex


    def make_clusters(self):

        try:
            self.log_writer.log(self.file_object, "Clustering has been initiated")
            find_clusters = self.find_clusters()
            no_of_cluster = find_clusters                              #Get number of clusters
            self.clusters = KMeans(no_of_cluster, random_state=30)     #object initialisation for Kmeans
            self.yclus = self.clusters.fit_predict(self.cdata)         #Get the array of cluster numbers
            #pickle.dump(self.clusters, open('CLUSTER.sav', 'wb'))      #Save the cluster model for prediction phase
            self.log_writer.log(self.file_object, "{} clusters are made".format(no_of_cluster))
            #print(self.yclus)

            self.data['cluster_no'] = self.yclus     #Add cluster column to dataset
            print(self.data['cluster_no'])
            self.log_writer.log(self.file_object, "Cluster column is added to data")

            return self.clusters     #Return cluster model

        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured in clustering')
            raise ex
