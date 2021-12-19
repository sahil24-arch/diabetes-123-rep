import pandas as pd
from logger import App_Logger
from file_operation import FileOperations

class pred:

    def __init__(self,data):
        self.data=data
        self.file_opn=FileOperations()
        self.log_writer = App_Logger()
        self.file_object = open("All_logs/prediction_log.txt", 'a+')
        self.file_opn = FileOperations()

    def predict(self):
        self.log_writer.log(self.file_object, "Prediction is Initiated")

        load_cluster = self.file_opn.load_model('CLUSTER.sav')


        d=self.data.drop('Patient',axis=1)
        cluster_no = load_cluster.fit_predict(d)
        self.data['cluster_no']=cluster_no
        self.log_writer.log(self.file_object, "Clustering of Prediction data done")
        final_op=pd.DataFrame()

        for i in range(len(self.data['cluster_no'].unique())):
            self.log_writer.log(self.file_object, "Prediction for CLUSTER{}".format(i))
            cluster_data = self.data[self.data['cluster_no'] == i]

            filename='finalized_model{}.sav'.format(i)
            model=self.file_opn.load_model(filename)
            self.log_writer.log(self.file_object, "finalised_model{} loaded for prediction".format(i))

            d=cluster_data.drop(['cluster_no','Patient'],axis=1)

            print(d.columns)

            ypred=model.predict(d)
            self.log_writer.log(self.file_object, "Prediction for cluster{} data done".format(i))
            cluster_data['prediction']=ypred
            final_op=final_op.append(cluster_data)


        final_op=final_op[['Patient','prediction']]
        print(final_op)
        final_op=pd.merge(self.data,final_op,on='Patient')
        final_op=final_op[['Patient','prediction']]


        final_op.to_csv('Prediction output.csv',index=False)
        self.log_writer.log(self.file_object, "Predicted output saved")

        return final_op.head().to_json(orient="records")





