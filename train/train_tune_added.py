# *****************************************************************
# Hyperparameters Optimization
# Script for different classifiers using Scikit-learn Library *****
# *****************************************************************

import json, argparse, pickle, os, sys
import matplotlib.pyplot as plt
from hypopt import GridSearch
from sklearn import metrics
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import numpy as np
import pandas as pd


class Model_Development:
    def __init__(self, train_data, valid_data, test_data, classifier):
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.classifier = classifier
        self.random_state = 42
        self.verbose = 2

        self.results = dict()
        
        self.output_dir = '../redial-2020-notebook-work/reports_tuned_best'
        self.output_models = '../redial-2020-notebook-work/models_tuned_best'
        
        if not os.path.isdir(self.output_dir): os.mkdir(self.output_dir)
        if not os.path.isdir(self.output_models): os.mkdir(self.output_models)

        _, file_name = os.path.split(test_data)
        self.file_name, _ = os.path.splitext(file_name)
        self.result_file = os.path.join(self.output_dir, f"{self.file_name}_results.json")
        self.file_name2 = self.file_name[:-3]
        self.model_file = os.path.join(self.output_models, f"{self.file_name2}.pkl")

    def evaluation_metrics(self, y_true, y_pred, y_prob):
        
        Scores = dict()
        
        roc_auc = metrics.roc_auc_score(y_true, y_prob)
        acc = metrics.accuracy_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred, average = 'binary')
        precision = metrics.precision_score(y_true, y_pred, average= 'binary')
        recall = metrics.recall_score(y_true, y_pred, average = 'binary')
        kappa = metrics.cohen_kappa_score(y_true, y_pred)
        
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        SP = float(tn)/(tn+fp)
        # Convert numpy.float64 to float using 'tolist()' and save the scores in dictonary 
        Scores['ACC'] = acc.tolist()
        Scores['F1_Score'] = f1.tolist()
        Scores['Recall'] = recall.tolist()
        Scores['Precision'] = precision.tolist()
        Scores['SP'] = SP.tolist()
        Scores['Cohen_Kappa'] = kappa.tolist()
        Scores['AUC'] = roc_auc.tolist()
        Scores['TP'] = tp.tolist()
        Scores['TN'] = tn.tolist()
        Scores['FP'] = fp.tolist()
        Scores['FN'] = fn.tolist()
        
        return Scores
        
    def get_data_sets(self):
        
        train_data = np.load(self.train_data)
        valid_data = np.load(self.valid_data)
        test_data = np.load(self.test_data)

        self.X_train = train_data[:, :-1]
        self.y_train = train_data[:, -1]
        self.X_test = test_data[:, :-1]
        self.y_test = test_data[:, -1]
        self.X_valid = valid_data[:, :-1]
        self.y_valid = valid_data[:, -1]

        return True

    def load_params(self):
        
        with open('hyperparams.json', 'r') as hyp:
            self.params = json.load(hyp)
        
        def get_hidden_layers():
            import itertools
            x = [64, 100]
            hl = []

            for i in range(1, len(x)):
                hl.extend([p for p in itertools.product(x, repeat=i+1)])

            return hl

        hidden_layer_sizes = get_hidden_layers()
        # params_grid for mlp will be updated !
        hidden_layer_sizes.append((100,))
        self.params['mlp'] = {
                                'hidden_layer_sizes': hidden_layer_sizes,
                                'max_iter':[100, 200, 300],
                                'random_state':[self.random_state],
                                'verbose':[self.verbose]
                            }
        return True

    def write_results(self):
        with open(self.result_file, 'w') as f:
            json.dump(self.results, f)
            
    def random_forest(self):
        model = RandomForestClassifier(random_state = self.random_state, n_jobs = -1)

        opt = GridSearch(model, param_grid = self.params['rf'], seed = self.random_state)

        opt.fit(self.X_train, self.y_train, self.X_valid, self.y_valid, verbose = self.verbose)
        output = open(self.model_file, 'wb')
        pickle.dump(opt, output)
        
        y_valid_pred = opt.predict(self.X_valid)
        y_valid_prob = opt.predict_proba(self.X_valid)[:,1]
        scores_valid = self.evaluation_metrics(self.y_valid, y_valid_pred, y_valid_prob)

        y_test_pred = opt.predict(self.X_test)
        y_test_prob = opt.predict_proba(self.X_test)[:,1]
        scores_test = self.evaluation_metrics(self.y_test, y_test_pred, y_test_prob)

        scores = dict()

        scores['valid'] = scores_valid
        scores['test'] = scores_test

        # Store the validation and test results

        self.results['RF'] = scores
    
    def decision_tree(self):

        model = DecisionTreeClassifier(random_state = self.random_state)

        opt = GridSearch(model, param_grid = self.params['dt'], seed = self.random_state)

        opt.fit(self.X_train, self.y_train, self.X_valid, self.y_valid, verbose = self.verbose)
        
        output = open(self.model_file, 'wb')
        pickle.dump(opt, output)

        y_valid_pred = opt.predict(self.X_valid)
        y_valid_prob = opt.predict_proba(self.X_valid)[:,1]
        scores_valid = self.evaluation_metrics(self.y_valid, y_valid_pred, y_valid_prob)

        y_test_pred = opt.predict(self.X_test)
        y_test_prob = opt.predict_proba(self.X_test)[:,1]
        scores_test = self.evaluation_metrics(self.y_test, y_test_pred, y_test_prob)

        scores = dict()

        scores['valid'] = scores_valid
        scores['test'] = scores_test

        # Store the validation and test results

        self.results['DT'] = scores

    def ada_boost(self):
        
        model = AdaBoostClassifier(random_state = self.random_state)

        opt = GridSearch(model, param_grid = self.params['ada'], seed = self.random_state)

        opt.fit(self.X_train, self.y_train, self.X_valid, self.y_valid, verbose = self.verbose)
        output = open(self.model_file, 'wb')
        pickle.dump(opt, output)

        y_valid_pred = opt.predict(self.X_valid)
        y_valid_prob = opt.predict_proba(self.X_valid)[:,1]
        scores_valid = self.evaluation_metrics(self.y_valid, y_valid_pred, y_valid_prob)

        y_test_pred = opt.predict(self.X_test)
        y_test_prob = opt.predict_proba(self.X_test)[:,1]
        scores_test = self.evaluation_metrics(self.y_test, y_test_pred, y_test_prob)

        scores = dict()

        scores['valid'] = scores_valid
        scores['test'] = scores_test

        # Store the validation and test results

        self.results['ADA'] = scores
    
    def logistic_regression(self):    
        
        model = LogisticRegression(random_state = self.random_state)

        opt = GridSearch(model, param_grid = self.params['lr'], seed = self.random_state)

        opt.fit(self.X_train, self.y_train, self.X_valid, self.y_valid, verbose = self.verbose)
        output = open(self.model_file, 'wb')
        pickle.dump(opt, output)

        y_valid_pred = opt.predict(self.X_valid)
        y_valid_prob = opt.predict_proba(self.X_valid)[:,1]
        scores_valid = self.evaluation_metrics(self.y_valid, y_valid_pred, y_valid_prob)

        y_test_pred = opt.predict(self.X_test)
        y_test_prob = opt.predict_proba(self.X_test)[:,1]
        scores_test = self.evaluation_metrics(self.y_test, y_test_pred, y_test_prob)

        scores = dict()

        scores['valid'] = scores_valid
        scores['test'] = scores_test

        # Store the validation and test results

        self.results['LR'] = scores
    
    def xgb(self):

        model = XGBClassifier(random_state = self.random_state)

        opt = GridSearch(model, param_grid = self.params['xgb'], seed = self.random_state)

        opt.fit(self.X_train, self.y_train, self.X_valid, self.y_valid, verbose = self.verbose)
        output = open(self.model_file, 'wb')
        pickle.dump(opt, output)

        y_valid_pred = opt.predict(self.X_valid)
        y_valid_prob = opt.predict_proba(self.X_valid)[:,1]
        scores_valid = self.evaluation_metrics(self.y_valid, y_valid_pred, y_valid_prob)

        y_test_pred = opt.predict(self.X_test)
        y_test_prob = opt.predict_proba(self.X_test)[:,1]
        scores_test = self.evaluation_metrics(self.y_test, y_test_pred, y_test_prob)

        scores = dict()

        scores['valid'] = scores_valid
        scores['test'] = scores_test

        # Store the validation and test results

        self.results['XGB'] = scores
    
    def multinomial_nb(self):

        model = MultinomialNB() # random_state not present
        scores = dict()
        
        try:
            opt = GridSearch(model, param_grid = self.params['mnb'], seed = self.random_state)

            opt.fit(self.X_train, self.y_train, self.X_valid, self.y_valid, verbose = self.verbose)
            output = open(self.model_file, 'wb')
            pickle.dump(opt, output)
            y_valid_pred = opt.predict(self.X_valid)
            y_valid_prob = opt.predict_proba(self.X_valid)[:,1]
            scores_valid = self.evaluation_metrics(self.y_valid, y_valid_pred, y_valid_prob)

            y_test_pred = opt.predict(self.X_test)
            y_test_prob = opt.predict_proba(self.X_test)[:,1]
            scores_test = self.evaluation_metrics(self.y_test, y_test_pred, y_test_prob)
            scores['valid'] = scores_valid
            scores['test'] = scores_test
        except:
            scores['valid'] = 'NaN'
            scores['test'] = 'NaN'
        
        # Store the validation and test results

        self.results['MNB'] = scores
    
    def gaussian_nb(self):

        model = GaussianNB() # random_state not present

        opt = GridSearch(model, param_grid = self.params['gnb'], seed = self.random_state)

        opt.fit(self.X_train, self.y_train, self.X_valid, self.y_valid, verbose = self.verbose)
        output = open(self.model_file, 'wb')
        pickle.dump(opt, output)

        y_valid_pred = opt.predict(self.X_valid)
        y_valid_prob = opt.predict_proba(self.X_valid)[:,1]
        scores_valid = self.evaluation_metrics(self.y_valid, y_valid_pred, y_valid_prob)

        y_test_pred = opt.predict(self.X_test)
        y_test_prob = opt.predict_proba(self.X_test)[:,1]
        scores_test = self.evaluation_metrics(self.y_test, y_test_pred, y_test_prob)

        scores = dict()

        scores['valid'] = scores_valid
        scores['test'] = scores_test

        # Store the validation and test results

        self.results['GNB'] = scores
     
    def kneighbors(self):
        
        model = KNeighborsClassifier() # random_state not present.

        opt = GridSearch(model, param_grid = self.params['knb'], seed = self.random_state)

        opt.fit(self.X_train, self.y_train, self.X_valid, self.y_valid, verbose = self.verbose)
        output = open(self.model_file, 'wb')
        pickle.dump(opt, output)

        y_valid_pred = opt.predict(self.X_valid)
        y_valid_prob = opt.predict_proba(self.X_valid)[:,1]
        scores_valid = self.evaluation_metrics(self.y_valid, y_valid_pred, y_valid_prob)

        y_test_pred = opt.predict(self.X_test)
        y_test_prob = opt.predict_proba(self.X_test)[:,1]
        scores_test = self.evaluation_metrics(self.y_test, y_test_pred, y_test_prob)

        scores = dict()

        scores['valid'] = scores_valid
        scores['test'] = scores_test

        # Store the validation and test results

        self.results['KNB'] = scores
    
    def dummy(self):

        model = DummyClassifier(random_state = self.random_state)

        opt = GridSearch(model, param_grid = self.params['dum'], seed = self.random_state)

        opt.fit(self.X_train, self.y_train, self.X_valid, self.y_valid, verbose = self.verbose)
        output = open(self.model_file, 'wb')
        pickle.dump(opt, output)

        y_valid_pred = opt.predict(self.X_valid)
        y_valid_prob = opt.predict_proba(self.X_valid)[:,1]
        scores_valid = self.evaluation_metrics(self.y_valid, y_valid_pred, y_valid_prob)

        y_test_pred = opt.predict(self.X_test)
        y_test_prob = opt.predict_proba(self.X_test)[:,1]
        scores_test = self.evaluation_metrics(self.y_test, y_test_pred, y_test_prob)

        scores = dict()

        scores['valid'] = scores_valid
        scores['test'] = scores_test

        # Store the validation and test results

        self.results['DUM'] = scores
    
    
    def mlp(self):

        model = MLPClassifier(random_state = self.random_state)

        opt = GridSearch(model, param_grid = self.params['mlp'], seed = self.random_state)

        opt.fit(self.X_train, self.y_train, self.X_valid, self.y_valid, verbose = self.verbose)
        output = open(self.model_file, 'wb')
        pickle.dump(opt, output)

        y_valid_pred = opt.predict(self.X_valid)
        y_valid_prob = opt.predict_proba(self.X_valid)[:,1]
        scores_valid = self.evaluation_metrics(self.y_valid, y_valid_pred, y_valid_prob)

        y_test_pred = opt.predict(self.X_test)
        y_test_prob = opt.predict_proba(self.X_test)[:,1]
        scores_test = self.evaluation_metrics(self.y_test, y_test_pred, y_test_prob)

        scores = dict()

        scores['valid'] = scores_valid
        scores['test'] = scores_test

        # Store the validation and test results

        self.results['MLP'] = scores

    
    def svc(self):
        
        model = SVC(probability = True, random_state = self.random_state)

        opt = GridSearch(model, param_grid = self.params['svc'], seed = self.random_state)

        opt.fit(self.X_train, self.y_train, self.X_valid, self.y_valid, verbose = self.verbose)
        output = open(self.model_file, 'wb')
        pickle.dump(opt, output)

        y_valid_pred = opt.predict(self.X_valid)
        y_valid_prob = opt.predict_proba(self.X_valid)[:,1]
        scores_valid = self.evaluation_metrics(self.y_valid, y_valid_pred, y_valid_prob)

        y_test_pred = opt.predict(self.X_test)
        y_test_prob = opt.predict_proba(self.X_test)[:,1]
        scores_test = self.evaluation_metrics(self.y_test, y_test_pred, y_test_prob)

        scores = dict()

        scores['valid'] = scores_valid
        scores['test'] = scores_test

        # Store the validation and test results

        self.results['SVC'] = scores
    
    def nu_svc(self):
        
        model = NuSVC(random_state = self.random_state)

        opt = GridSearch(model, param_grid = self.params['nusvc'], seed = self.random_state)

        opt.fit(self.X_train, self.y_train, self.X_valid, self.y_valid, verbose = self.verbose)
        output = open(self.model_file, 'wb')
        pickle.dump(opt, output)

        y_valid_pred = opt.predict(self.X_valid)
        y_valid_prob = opt.predict_proba(self.X_valid)[:,1]
        scores_valid = self.evaluation_metrics(self.y_valid, y_valid_pred, y_valid_prob)

        y_test_pred = opt.predict(self.X_test)
        y_test_prob = opt.predict_proba(self.X_test)[:,1]
        scores_test = self.evaluation_metrics(self.y_test, y_test_pred, y_test_prob)

        scores = dict()

        scores['valid'] = scores_valid
        scores['test'] = scores_test

        # Store the validation and test results

        self.results['NuSVC'] = scores
    
    
    def gradient_boosting(self):
        
        model = GradientBoostingClassifier(random_state = self.random_state)

        opt = GridSearch(model, param_grid = self.params['gb'], seed = self.random_state)

        opt.fit(self.X_train, self.y_train, self.X_valid, self.y_valid, verbose = self.verbose)

        output = open(self.model_file, 'wb')
        pickle.dump(opt, output)

        y_valid_pred = opt.predict(self.X_valid)
        y_valid_prob = opt.predict_proba(self.X_valid)[:,1]
        scores_valid = self.evaluation_metrics(self.y_valid, y_valid_pred, y_valid_prob)

        y_test_pred = opt.predict(self.X_test)
        y_test_prob = opt.predict_proba(self.X_test)[:,1]
        scores_test = self.evaluation_metrics(self.y_test, y_test_pred, y_test_prob)

        scores = dict()

        scores['valid'] = scores_valid
        scores['test'] = scores_test

        # Store the validation and test results

        self.results['GB'] = scores
    
    def linear_discriminant_analysis(self):

        model = LinearDiscriminantAnalysis(random_state = self.random_state)

        opt = GridSearch(model, param_grid = self.params['lda'], seed = self.random_state)

        output = open(self.model_file, 'wb')
        pickle.dump(opt, output)

        opt.fit(self.X_train, self.y_train, self.X_valid, self.y_valid, verbose = self.verbose)

        y_valid_pred = opt.predict(self.X_valid)
        y_valid_prob = opt.predict_proba(self.X_valid)[:,1]
        scores_valid = self.evaluation_metrics(self.y_valid, y_valid_pred, y_valid_prob)

        y_test_pred = opt.predict(self.X_test)
        y_test_prob = opt.predict_proba(self.X_test)[:,1]
        scores_test = self.evaluation_metrics(self.y_test, y_test_pred, y_test_prob)

        scores = dict()

        scores['valid'] = scores_valid
        scores['test'] = scores_test

        # Store the validation and test results

        self.results['LDA'] = scores
    
    
    def quadratic_discriminant_analysis(self):
    
        model = QuadraticDiscriminantAnalysis(random_state = self.random_state)

        opt = GridSearch(model, param_grid = self.params['qda'], seed = self.random_state)

        opt.fit(self.X_train, self.y_train, self.X_valid, self.y_valid, verbose = self.verbose)

        output = open(self.model_file, 'wb')
        pickle.dump(opt, output)

        y_valid_pred = opt.predict(self.X_valid)
        y_valid_prob = opt.predict_proba(self.X_valid)[:,1]
        scores_valid = self.evaluation_metrics(self.y_valid, y_valid_pred, y_valid_prob)

        y_test_pred = opt.predict(self.X_test)
        y_test_prob = opt.predict_proba(self.X_test)[:,1]
        scores_test = self.evaluation_metrics(self.y_test, y_test_pred, y_test_prob)

        scores = dict()

        scores['valid'] = scores_valid
        scores['test'] = scores_test

        # Store the validation and test results

        self.results['QDA'] = scores
        
    def extra_trees(self):

        model = ExtraTreesClassifier(random_state = self.random_state)

        opt = GridSearch(model, param_grid = self.params['etas'], seed = self.random_state)

        opt.fit(self.X_train, self.y_train, self.X_valid, self.y_valid, verbose = self.verbose)
        
        output = open(self.model_file, 'wb')
        pickle.dump(opt, output)

        y_valid_pred = opt.predict(self.X_valid)
        y_valid_prob = opt.predict_proba(self.X_valid)[:,1]
        scores_valid = self.evaluation_metrics(self.y_valid, y_valid_pred, y_valid_prob)

        y_test_pred = opt.predict(self.X_test)
        y_test_prob = opt.predict_proba(self.X_test)[:,1]
        scores_test = self.evaluation_metrics(self.y_test, y_test_pred, y_test_prob)

        scores = dict()

        scores['valid'] = scores_valid
        scores['test'] = scores_test

        self.results['ETAs'] = scores
    
    def extra_tree(self):

        model = ExtraTreeClassifier(random_state = self.random_state)

        opt = GridSearch(model, param_grid = self.params['etas'], seed = self.random_state)

        opt.fit(self.X_train, self.y_train, self.X_valid, self.y_valid, verbose = self.verbose)

        output = open(self.model_file, 'wb')
        pickle.dump(opt, output)

        y_valid_pred = opt.predict(self.X_valid)
        y_valid_prob = opt.predict_proba(self.X_valid)[:,1]
        scores_valid = self.evaluation_metrics(self.y_valid, y_valid_pred, y_valid_prob)

        y_test_pred = opt.predict(self.X_test)
        y_test_prob = opt.predict_proba(self.X_test)[:,1]
        scores_test = self.evaluation_metrics(self.y_test, y_test_pred, y_test_prob)

        scores = dict()

        scores['valid'] = scores_valid
        scores['test'] = scores_test

        self.results['ETA'] = scores

    def output_code(self):
        
        model = OutputCodeClassifier(MultinomialNB(), random_state = self.random_state)

        opt = GridSearch(model, param_grid = self.params['out'], seed = self.random_state)

        output = open(self.model_file, 'wb')
        pickle.dump(opt, output)

        opt.fit(self.X_train, self.y_train, self.X_valid, self.y_valid, verbose = self.verbose)

        y_valid_pred = opt.predict(self.X_valid)
        y_valid_prob = opt.predict_proba(self.X_valid)[:,1]
        scores_valid = self.evaluation_metrics(self.y_valid, y_valid_pred, y_valid_prob)

        y_test_pred = opt.predict(self.X_test)
        y_test_prob = opt.predict_proba(self.X_test)[:,1]
        scores_test = self.evaluation_metrics(self.y_test, y_test_pred, y_test_prob)

        scores = dict()

        scores['valid'] = scores_valid
        scores['test'] = scores_test

        self.results['OUT'] = scores 


    def oneVsrest(self):
        
        model = OneVsRestClassifier(LogisticRegression(random_state=self.random_state))

        opt = GridSearch(model, param_grid = self.params['ovr'], seed = self.random_state)

        opt.fit(self.X_train, self.y_train, self.X_valid, self.y_valid, verbose = self.verbose)

        output = open(self.model_file, 'wb')
        pickle.dump(opt, output)

        y_valid_pred = opt.predict(self.X_valid)
        y_valid_prob = opt.predict_proba(self.X_valid)[:,1]
        scores_valid = self.evaluation_metrics(self.y_valid, y_valid_pred, y_valid_prob)

        y_test_pred = opt.predict(self.X_test)
        y_test_prob = opt.predict_proba(self.X_test)[:,1]
        scores_test = self.evaluation_metrics(self.y_test, y_test_pred, y_test_prob)

        scores = dict()

        scores['valid'] = scores_valid
        scores['test'] = scores_test

        self.results['OVR'] = scores

    def oneVsone(self):
        
        model = OneVsOneClassifier(MultinomialNB()) # random_state not present

        opt = GridSearch(model, param_grid = self.params['ovo'], seed = self.random_state)

        opt.fit(self.X_train, self.y_train, self.X_valid, self.y_valid, verbose = self.verbose)

        output = open(self.model_file, 'wb')
        pickle.dump(opt, output)

        y_valid_pred = opt.predict(self.X_valid)
        y_valid_prob = opt.predict_proba(self.X_valid)[:,1]
        scores_valid = self.evaluation_metrics(self.y_valid, y_valid_pred, y_valid_prob)

        y_test_pred = opt.predict(self.X_test)
        y_test_prob = opt.predict_proba(self.X_test)[:,1]
        scores_test = self.evaluation_metrics(self.y_test, y_test_pred, y_test_prob)

        scores = dict()

        scores['valid'] = scores_valid
        scores['test'] = scores_test

        self.results['OVO'] = scores

    def ridge(self):
        
        model = RidgeClassifier(random_state = self.random_state)

        opt = GridSearch(model, param_grid = self.params['ridge'], seed = self.random_state)

        opt.fit(self.X_train, self.y_train, self.X_valid, self.y_valid, verbose = self.verbose)
        output = open(self.model_file, 'wb')
        pickle.dump(opt, output)

        y_valid_pred = opt.predict(self.X_valid)
        y_valid_prob = opt.predict_proba(self.X_valid)[:,1]
        scores_valid = self.evaluation_metrics(self.y_valid, y_valid_pred, y_valid_prob)

        y_test_pred = opt.predict(self.X_test)
        y_test_prob = opt.predict_proba(self.X_test)[:,1]
        scores_test = self.evaluation_metrics(self.y_test, y_test_pred, y_test_prob)

        scores = dict()

        scores['valid'] = scores_valid
        scores['test'] = scores_test

        self.results['RIDGE'] = scores

    def nearest_centroid(self):
        
        model = NearestCentroid(random_state = self.random_state)

        opt = GridSearch(model, param_grid = self.params['nr'], seed = self.random_state)

        opt.fit(self.X_train, self.y_train, self.X_valid, self.y_valid, verbose = self.verbose)
        output = open(self.model_file, 'wb')
        pickle.dump(opt, output)

        y_valid_pred = opt.predict(self.X_valid)
        y_valid_prob = opt.predict_proba(self.X_valid)[:,1]
        scores_valid = self.evaluation_metrics(self.y_valid, y_valid_pred, y_valid_prob)

        y_test_pred = opt.predict(self.X_test)
        y_test_prob = opt.predict_proba(self.X_test)[:,1]
        scores_test = self.evaluation_metrics(self.y_test, y_test_pred, y_test_prob)

        scores = dict()

        scores['valid'] = scores_valid
        scores['test'] = scores_test

        self.results['NR'] = scores


    def sgd(self):
        
        model = SGDClassifier(random_state = self.random_state)

        opt = GridSearch(model, param_grid = self.params['sgd'], seed = self.random_state)
        output = open(self.model_file, 'wb')
        pickle.dump(opt, output)

        opt.fit(self.X_train, self.y_train, self.X_valid, self.y_valid, verbose = self.verbose)

        y_valid_pred = opt.predict(self.X_valid)
        y_valid_prob = opt.predict_proba(self.X_valid)[:,1]
        scores_valid = self.evaluation_metrics(self.y_valid, y_valid_pred, y_valid_prob)

        y_test_pred = opt.predict(self.X_test)
        y_test_prob = opt.predict_proba(self.X_test)[:,1]
        scores_test = self.evaluation_metrics(self.y_test, y_test_pred, y_test_prob)

        scores = dict()

        scores['valid'] = scores_valid
        scores['test'] = scores_test

        self.results['SGD'] = scores

    def passive_aggressive(self):
        
        model = PassiveAggressiveClassifier(random_state = self.random_state)

        opt = GridSearch(model, param_grid = self.params['pac'], seed = self.random_state)

        opt.fit(self.X_train, self.y_train, self.X_valid, self.y_valid, verbose = self.verbose)

        output = open(self.model_file, 'wb')
        pickle.dump(opt, output)

        y_valid_pred = opt.predict(self.X_valid)
        y_valid_prob = opt.predict_proba(self.X_valid)[:,1]
        scores_valid = self.evaluation_metrics(self.y_valid, y_valid_pred, y_valid_prob)

        y_test_pred = opt.predict(self.X_test)
        y_test_prob = opt.predict_proba(self.X_test)[:,1]
        scores_test = self.evaluation_metrics(self.y_test, y_test_pred, y_test_prob)

        scores = dict()

        scores['valid'] = scores_valid
        scores['test'] = scores_test

        self.results['PAC'] = scores
        
    def complement_nb(self):
        
        model = ComplementNB() # random_state not present
            
        scores = dict()
        
        try:
            opt = GridSearch(model, param_grid = self.params['cnb'], seed = self.random_state)

            opt.fit(self.X_train, self.y_train, self.X_valid, self.y_valid, verbose = self.verbose)
            output = open(self.model_file, 'wb')
            pickle.dump(opt, output) 

            y_valid_pred = opt.predict(self.X_valid)
            y_valid_prob = opt.predict_proba(self.X_valid)[:,1]
            scores_valid = self.evaluation_metrics(self.y_valid, y_valid_pred, y_valid_prob)

            y_test_pred = opt.predict(self.X_test)
            y_test_prob = opt.predict_proba(self.X_test)[:,1]
            scores_test = self.evaluation_metrics(self.y_test, y_test_pred, y_test_prob)

            scores['valid'] = scores_valid
            scores['test'] = scores_test
        except:

            scores['valid'] = "NaN"
            scores['test'] = "NaN"

        self.results['CNB'] = scores


    def linear_svc(self):
        
        model = LinearSVC(random_state = self.random_state)

        opt = GridSearch(model, param_grid = self.params['lsvc'], seed = self.random_state)

        opt.fit(self.X_train, self.y_train, self.X_valid, self.y_valid, verbose = self.verbose)
        
        output = open(self.model_file, 'wb')
        pickle.dump(opt, output)

        y_valid_pred = opt.predict(self.X_valid)
        y_valid_prob = opt.predict_proba(self.X_valid)[:,1]
        scores_valid = self.evaluation_metrics(self.y_valid, y_valid_pred, y_valid_prob)

        y_test_pred = opt.predict(self.X_test)
        y_test_prob = opt.predict_proba(self.X_test)[:,1]
        scores_test = self.evaluation_metrics(self.y_test, y_test_pred, y_test_prob)

        scores = dict()

        scores['valid'] = scores_valid
        scores['test'] = scores_test

        self.results['LSVC'] = scores

    def perceptron(self):
        
        model = Perceptron(random_state = self.random_state)

        opt = GridSearch(model, param_grid = self.params['per'], seed = self.random_state)

        opt.fit(self.X_train, self.y_train, self.X_valid, self.y_valid, verbose = self.verbose)
        output = open(self.model_file, 'wb')
        pickle.dump(opt, output)
        
        y_valid_pred = opt.predict(self.X_valid)
        y_valid_prob = opt.predict_proba(self.X_valid)[:,1]
        scores_valid = self.evaluation_metrics(self.y_valid, y_valid_pred, y_valid_prob)

        y_test_pred = opt.predict(self.X_test)
        y_test_prob = opt.predict_proba(self.X_test)[:,1]
        scores_test = self.evaluation_metrics(self.y_test, y_test_pred, y_test_prob)

        scores = dict()

        scores['valid'] = scores_valid
        scores['test'] = scores_test

        self.results['PER'] = scores


    def bagging(self):
        
        model = BaggingClassifier(random_state = self.random_state)

        opt = GridSearch(model, param_grid = self.params['bag'], seed = self.random_state)

        opt.fit(self.X_train, self.y_train, self.X_valid, self.y_valid, verbose = self.verbose)
        output = open(self.model_file, 'wb')
        pickle.dump(opt, output)

        y_valid_pred = opt.predict(self.X_valid)
        y_valid_prob = opt.predict_proba(self.X_valid)[:,1]
        scores_valid = self.evaluation_metrics(self.y_valid, y_valid_pred, y_valid_prob)

        y_test_pred = opt.predict(self.X_test)
        y_test_prob = opt.predict_proba(self.X_test)[:,1]
        scores_test = self.evaluation_metrics(self.y_test, y_test_pred, y_test_prob)
        
        scores = dict()
        scores['valid'] = scores_valid
        scores['test'] = scores_test

        self.results['BAG'] = scores


def main(classifier):

    md.get_data_sets()
    md.load_params()

    if classifier == 'OVR':
        md.oneVsrest()
    elif classifier == 'KNB':
        md.kneighbors()
    elif classifier == 'DT':
        md.decision_tree()
    elif classifier == 'MNB':
        md.multinomial_nb()
    elif classifier == 'RF':
        md.random_forest()
    elif classifier == 'ADA':
        md.ada_boost()
    elif classifier == 'XGB':
        md.xgb()
    elif classifier == 'MLP':
        md.mlp()
    elif classifier == 'LR':
        md.logistic_regression()
    elif classifier == 'ETAs':
        md.extra_trees()
    elif classifier == 'CNB':
        md.complement_nb()
    elif classifier == 'GB':
        md.gradient_boosting()
    elif classifier == 'BAG':
        md.bagging()
    elif classifier == 'SVC':
        md.svc()
    md.write_results()

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description="Training models")
    
    parser.add_argument('--train-features', action='store', dest='train-features', required=True, \
                        help='train-features file (numpyFile)')
    parser.add_argument('--validation-features', action='store', dest = 'validation-features', required=True,\
                        help='validation-features file (numpyFile)')
    parser.add_argument('--test-features', action='store', dest = 'test-features', required=True,\
                        help='test-features file (numpyFile)')
    parser.add_argument('--classifier', action='store', dest = 'classifier', required=True, help='type of classifier (sklearn)')

    args = vars(parser.parse_args())
    
    train = args['train-features']
    valid = args['validation-features']
    test = args['test-features']
    classifier = args['classifier']

    md = Model_Development(train, valid, test, classifier)

    main(classifier)

