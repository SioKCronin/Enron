#!/usr/bin/python

import sys
import numpy
import pickle
import matplotlib.pyplot
sys.path.append("../../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# Define functions

def sum_financial_fields(data):
    for key in data:
        fields = [ 'salary', 'bonus']
        total = 0
        for y in fields:
            if data[key][y] != 'NaN':
                total = total + data[key][y]
        data[key]['financial'] = total
    return data

def outlierCleaner(data, features):
    include = {}
    for key in data:
        if data[key]['financial'] in features:
            include[key] = data[key]
    return include

def feature_array(data, position):
    output = []
    for x in data:
        output.append(x[position])
    return output

def NAN_detector(array):
    count = 0
    for x in array:
        for y in x:
            if y == 'NaN':
                count = count + 1

def poi_detector(data):
    count = 0
    for key in data:
        if data[key]['poi'] == True:
            count += 1
    return count

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Create new feature
data_dict = sum_financial_fields(data_dict)

print "Original length:", len(data_dict)
print "POIs:", poi_detector(data_dict)
print len(data_dict['TOTAL'].keys())

### Plotting

# features = ["poi", "financial"]
# data_dict.pop('TOTAL', 0)
# data = featureFormat(data_dict, features)

# for point in data:
#     salary = point[0]
#     financial = point[1]
#     matplotlib.pyplot.scatter( salary, financial )
#
# matplotlib.pyplot.xlabel("poi")
# matplotlib.pyplot.ylabel("financial")
# matplotlib.pyplot.title('Before transform plot')
# matplotlib.pyplot.show()

### Plot regression to identify outliers

# from sklearn import linear_model
#
# financial = feature_array(features, 0)
# salary = feature_array(features, 1)
# financial       = numpy.reshape( numpy.array(financial), (len(financial), 1))
# salary = numpy.reshape( numpy.array(salary), (len(salary), 1))
#
# reg = linear_model.LinearRegression()
# reg.fit (salary, financial)
#
# try:
#     matplotlib.pyplot.plot(salary, reg.predict(salary), color="blue")
# except NameError:
#     pass
# matplotlib.pyplot.scatter(salary, financial)
# matplotlib.pyplot.show()

### Remove outliers

data_dict.pop( 'TOTAL', 0 )
data_dict.pop( 'THE TRAVEL AGENCY IN THE PARK', 0 )

print "New length:", len(data_dict)
print "NANs:", NAN_detector(data_dict)

### Extract features and labels from dataset for local testing
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = [ 'poi', 'financial', 'director_fees']

# features_list = [ 'poi', 'financial', 'director_fees', 'salary', 'bonus', 'total_stock_value']
data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Define classifiers

# Test four algorithms out of the box

from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print "GaussianNB accuracy:", accuracy_score(pred, labels_test)

from sklearn.svm import SVC
clf = SVC()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print "SVC:", accuracy_score(pred, labels_test)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print "Decision Tree:", accuracy_score(pred, labels_test)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print "KNearestNeighbors:", accuracy_score(pred, labels_test)

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
kbest = SelectKBest(f_classif)

# Define pipeline

# Scale and estimate with KNeighborsClassifier
pipe_knn = Pipeline(steps = [
    ('scale', MinMaxScaler()),
    ('estimate', KNeighborsClassifier(
        n_neighbors=15,
        weights='uniform',
        metric = 'minkowski'))])

# Compare with SVC
pipe_svc = Pipeline(steps = [
    ('scale', MinMaxScaler()),
    ('estimate', SVC())])

# Define parameters
params_knn = {'estimate__n_neighbors': [2, 5, 10]}
params_svc = {'estimate__kernel':('linear', 'rbf'), 'estimate__C':[1, 10]}

from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(
    n_splits=10,
    test_size=0.1,
    train_size=None,
    random_state= None
    )

clf = GridSearchCV(
    pipe_knn,
    param_grid = params_knn,
    scoring = 'f1',
    n_jobs = 1,
    cv = sss,
    verbose = 1,
    error_score = 0
    )

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, data_dict, features_list)
