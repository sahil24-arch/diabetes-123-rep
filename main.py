from wsgiref import simple_server
from flask import Flask, request, render_template
from flask import Response
import os
from flask_cors import CORS, cross_origin
import flask_monitoringdashboard as dashboard
import json
from model_training import train
from DataPreprocessing import scale_data
from prediction import pred

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
dashboard.bind(app)
CORS(app)


@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRouteClient():
    path = 'Training_files/diabetes_dataset'
    dp = scale_data(path)     #object initialisation
    data = dp.train_scale()   #get scaled data

    train1 = train(data)      #object initialisation
    train1.train_model()      #to start training;

    return Response('Training Done')


@app.route("/predict", methods=['GET', 'POST'])
@cross_origin()
def predictRouteClient():
    try:
        if request.json is not None:
            path = request.json['filepath']
            #path = 'Prediction_files/Diabetes_pred_data.csv'
            print(path)
            dp = scale_data(path)
            data = dp.pred_scale()
            # print(data.columns)

            pred1 = pred(data)
            json_predictions=pred1.predict()

            return Response('and few of the predictions are ' + str(
                json.loads(json_predictions)))

        elif request.form is not None:
            path = request.form['filepath']
            print(path)
            dp = scale_data(path)
            data = dp.pred_scale()
            print(data.columns)
            print(data)

            pred1 = pred(data)
            json_predictions = pred1.predict()

            return Response('and few of the predictions are ' + str(
                json.loads(json_predictions)))
    except Exception as e:
        raise e


# def predict():
#
#     #inputs.....
#
#     # load_cluster = pickle.load(open('CLUSTER.sav', 'rb'))
#     # cluster_no= load_cluster.fit_predict([[]])
#     # load_model= pickle.load(open('finalized_model{}.sav'.format(cluster_no), 'rb'))
#     # load_model.predict([[]])
#
#     path='Prediction_files/Diabetes_pred_data.csv'
#     dp=scale_data(path)
#     data=dp.pred_scale()
#     # print(data.columns)
#
#     pred1=pred(data)
#     pred1.predict()



port = int(os.getenv("PORT", 5000))
if __name__ == "__main__":
    host = '0.0.0.0'
    # port = 5000
    httpd = simple_server.make_server(host, port, app)
    # print("Serving on %s %d" % (host, port))
    httpd.serve_forever()

