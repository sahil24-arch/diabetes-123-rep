from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
# from sklearn.naive_bayes import BaseNB ,BernoulliNB ,CategoricalNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,roc_auc_score,precision_score,recall_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from logger import App_Logger

class best_model:

    """
    This class is used to find the best model out of Logistic Regression,RandomForestClassifier ,XGBoost CLassifier
    """

    def __init__(self, xtrain, xtest, ytrain, ytest):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest
        self.rf = RandomForestClassifier()
        self.lr = LogisticRegression()
        self.xgb = XGBClassifier(objective='binary:logistic')
        self.log_writer = App_Logger()
        self.file_object = open('All_logs/train_log.txt', 'a+')
        self.score_file = open('scores.txt','a+')
        print('training started!!!!')

    def get_best_parameters_rf(self):

        try:
            #Dictionery for hyperparameters to be tuned
            param_grid = {"n_estimators": [10, 50, 100, 130, 150, 200, 220], "criterion": ['gini', 'entropy'],
                          "max_depth": range(5, 41, 10),
                          "max_features": ['auto', 'log2'], 'min_samples_split': [x / 1000 for x in list(range(5, 41, 10))],
                          "min_samples_leaf": [x / 1000 for x in list(range(5, 41, 10))]}


            self.log_writer.log(self.file_object, 'Hyper parameter tuning of RandomForest Begun..')
            self.htune = GridSearchCV(self.rf, param_grid, cv=3, verbose=3,n_jobs=-1)  #initialise GridSearchCV
            self.htune.fit(self.xtrain, self.ytrain)     #Run hyperparameter tuning
            self.log_writer.log(self.file_object, 'Hyper parameter tuning of RandomForest Ended..')

            self.n_estimators = self.htune.best_params_['n_estimators']
            self.log_writer.log(self.file_object,'n_estimators ={}'.format(self.htune.best_params_['n_estimators']))

            self.criterion = self.htune.best_params_['criterion']
            self.log_writer.log(self.file_object,'criterion ={}'.format(self.htune.best_params_['criterion']))

            self.max_depth = self.htune.best_params_['max_depth']
            self.log_writer.log(self.file_object,'max_depth ={}'.format(self.htune.best_params_['max_depth']))

            self.max_features = self.htune.best_params_['max_features']
            self.log_writer.log(self.file_object,'max_features ={}'.format(self.htune.best_params_['max_features']))

            self.min_samples_split = self.htune.best_params_['min_samples_split']
            self.log_writer.log(self.file_object,'min_samples_split ={}'.format(self.htune.best_params_['min_samples_split']))

            self.min_samples_leaf = self.htune.best_params_['min_samples_leaf']
            self.log_writer.log(self.file_object,'min_samples_leaf ={}'.format(self.htune.best_params_['min_samples_leaf']))

            self.clf = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                              max_depth=self.max_depth, max_features=self.max_features)

            self.log_writer.log(self.file_object, 'Returning best Random Forest Classifier along with tuned parameters')

            return self.clf    #Returning RandomForest Classifier with best hyper parameters

        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured in hyperparameter tuning of Randomforest Classifier')
            raise ex

    def get_best_parameters_lr(self):

        try:
            # Dictionery for hyperparameters to be tuned
            param_grid = {"solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                              "penalty": ['l2', 'elasticnet'], "C": np.logspace(-4, 4, 10)}

            self.log_writer.log(self.file_object, 'Hyper parameter tuning of Logistic Regression Begun..')
            self.htune = GridSearchCV(self.lr, param_grid, cv=3, verbose=3, n_jobs=-1)   #initialise GridSearchCV
            self.htune.fit(self.xtrain, self.ytrain)
            self.log_writer.log(self.file_object, 'Hyper parameter tuning of Logistic Regression Ended..')

            self.solver = self.htune.best_params_['solver']
            self.log_writer.log(self.file_object, 'solver ={}'.format(self.htune.best_params_['solver']))

            self.penalty = self.htune.best_params_['penalty']
            self.log_writer.log(self.file_object, 'penalty ={}'.format(self.htune.best_params_['penalty']))

            self.C = self.htune.best_params_['C']
            self.log_writer.log(self.file_object, 'C ={}'.format(self.htune.best_params_['C']))

            self.clf = LogisticRegression(solver=self.solver, penalty=self.penalty, C=self.C)

            self.log_writer.log(self.file_object, 'Returning best Logistic Regressor along with tuned parameters')

            return self.clf          #Returning Logistic Regressor with best hyper parameters

        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured in hyperparameter tuning of Logistic Regressor')
            raise ex




    def get_best_parameters_xg(self):

        try:
            # Dictionery for hyperparameters to be tuned
            param_grid = {'learning_rate': [0.30, 0.25, 0.20, 0.15, 0.5, 0.1, 0.01, 0.001],
                          'max_depth': [3, 5, 6, 8, 10, 12, 15, 20],
                          'n_estimators': [10, 50, 100, 200],
                          'min_child_weight': [1, 3, 5, 7],
                          'gamma': [0.0, 0.1, 0.2, 0.3, 0.4],
                          'col_sample_bytree': [0.3, 0.4, 0.5, 0.7]}

            self.log_writer.log(self.file_object, 'Hyper parameter tuning of XGBoost Classifier Begun..')
            self.htune = GridSearchCV(XGBClassifier(objective='binary:logistic'), param_grid, cv=3, verbose=3,n_jobs=-1)
            self.htune.fit(self.xtrain, self.ytrain)
            self.log_writer.log(self.file_object, 'Hyper parameter tuning of XGBoost Classifier Ended..')

            self.learning_rate = self.htune.best_params_['learning_rate']
            self.log_writer.log(self.file_object,'learning_rate ={}'.format(self.htune.best_params_['learning_rate']))

            self.max_depth = self.htune.best_params_['max_depth']
            self.log_writer.log(self.file_object,'max_depth ={}'.format(self.htune.best_params_['max_depth']))

            self.n_estimators = self.htune.best_params_['n_estimators']
            self.log_writer.log(self.file_object,'n_estimators ={}'.format(self.htune.best_params_['n_estimators']))

            self.min_child_weight = self.htune.best_params_['min_child_weight']
            self.log_writer.log(self.file_object,'min_child_weight ={}'.format(self.htune.best_params_['min_child_weight']))

            self.gamma = self.htune.best_params_['gamma']
            self.log_writer.log(self.file_object,'gamma ={}'.format(self.htune.best_params_['gamma']))

            self.col_sample_bytree = self.htune.best_params_['col_sample_bytree']
            self.log_writer.log(self.file_object,'col_sample_bytree ={}'.format(self.htune.best_params_['col_sample_bytree']))

            self.clf = XGBClassifier(learning_rate=self.learning_rate, max_depth=self.max_depth,
                                     n_estimators=self.n_estimators,
                                     min_child_weight=self.min_child_weight, gamma=self.gamma,
                                     col_sample_bytree=self.col_sample_bytree)

            self.log_writer.log(self.file_object, 'Returning best XGBoost Classifier along with tuned parameters')

            return self.clf          #Returning XGBoost Classifier with best hyper parameters

        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured in hyperparameter tuning of XGBoost classifier Classifier')
            raise ex

    def model_finder(self):

        try:
            model_dict = {}  #Empty dictionery to save models along with auc acore

            # Logistic Regression

            self.log_writer.log(self.file_object,'Training Logistic Regression Model has begun')
            self.model1 = self.get_best_parameters_lr()     #Get best LR model

            self.model1.fit(self.xtrain, self.ytrain)        #Train with LR model
            self.log_writer.log(self.file_object, 'Training of best Logistic Regression Model is done ')

            self.ypred = self.model1.predict(self.xtest)    #get prediction from testing data
            score = roc_auc_score(self.ytest, self.ypred)     #Find auc score
            ac_score = accuracy_score(self.ytest, self.ypred)    #Find accuracy score
            p_score = precision_score(self.ytest, self.ypred)    #Find precission score
            r_score = recall_score(self.ytest, self.ypred)       #Find recall score
            self.log_writer.log(self.score_file,
                'Testing of Logistic Regression Model is done:: accuracy_score={}   precision_score={}   recall_score={}   auc-score={}'.format(ac_score, p_score, r_score,
                                                                                                 score))
            model_dict["lr"] = [score]      #append to dictionery

            # Random Forest Classifier
            self.log_writer.log(self.file_object,'Training RandomForest Classifier Model has begun')
            self.model2 = self.get_best_parameters_rf()         #Get best RF model

            self.model2.fit(self.xtrain, self.ytrain)           #Train with RF model
            self.log_writer.log(self.file_object, 'Training of best RandomForest Classifier Model is done ')

            self.ypred = self.model2.predict(self.xtest)      #get prediction from testing data
            score = roc_auc_score(self.ytest, self.ypred)     #Find auc score
            ac_score = accuracy_score(self.ytest, self.ypred) #Find accuracy score
            p_score = precision_score(self.ytest, self.ypred) #Find precission score
            r_score = recall_score(self.ytest, self.ypred)     #Find recall score
            self.log_writer.log(self.score_file,
                'Testing of RandomForest Classifier Model is done::accuracy_score={}   precision_score={}   recall_score={}   auc-score={}'.format(ac_score, p_score, r_score,
                                                                                                 score))
            model_dict["rf"] = score        #append to dictionery

            # xgboost
            self.log_writer.log(self.file_object,'Training of XGBoost Classifier Model has begun')
            self.model3 = self.get_best_parameters_rf()       #Get best XGB model

            self.model3.fit(self.xtrain, self.ytrain)         #Train with XGB model
            self.log_writer.log(self.file_object, 'Training of best XGBoost Classifier Model is done ')

            self.ypred = self.model3.predict(self.xtest)           #get prediction from testing data
            score = roc_auc_score(self.ytest, self.ypred)          #Find auc score
            ac_score = accuracy_score(self.ytest, self.ypred)      #Find accuracy score
            p_score = precision_score(self.ytest, self.ypred)      #Find precission score
            r_score = recall_score(self.ytest, self.ypred)         #Find recall score
            self.log_writer.log(self.score_file,
                'Testing of XGBoost Classifier Model is done::accuracy_score={}   precision_score={}   recall_score={}   auc-score={}'.format(ac_score, p_score, r_score,
                                                                                                 score))
            model_dict["xg"] = score        #append to dictionery

            key_list = list(model_dict.keys())         #get list of all keys(model name)
            value_list = list(model_dict.values())     #get list of all values(model auc scores)

            position = value_list.index(max(model_dict.values()))    #get the position of model with best auc score
            model = key_list[position]     #get the best model

            if key_list[position]=='lr':
                self.log_writer.log(self.file_object, 'Best model selected and returned::{}=={}'.format(key_list[position],value_list[position]))
                self.log_writer.log(self.score_file, 'Best model selected and returned::{}=={}'.format(key_list[position],value_list[position]))
                return self.model1       #Return Logist Regressor

            elif key_list[position]=='rf':
                self.log_writer.log(self.file_object, 'Best model selected and returned::{}=={}'.format(key_list[position],value_list[position]))
                self.log_writer.log(self.score_file, 'Best model selected and returned::{}=={}'.format(key_list[position],value_list[position]))
                return self.model2     #Return RandomForest CLassifier
            else:
                self.log_writer.log(self.file_object, 'Best model selected and returned::{}=={}'.format(key_list[position],value_list[position]))
                self.log_writer.log(self.score_file, 'Best model selected and returned::{}=={}'.format(key_list[position],value_list[position]))
                return self.model3      #Return XGBoost CLassifier

        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured in finding Best model')
            raise ex
