#!/usr/bin/python

import sys
import numpy
import pickle
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# Define functions

def sum_financial_fields(data):
    for key in data:
        fields = [ 'salary', 'bonus', 'total_stock_value']
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

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Create new feature
data_dict = sum_financial_fields(data_dict)
print "Original length:", len(data_dict)

### Plotting

# features = ["poi", "financial"]
# data_dict.pop('TOTAL', 0)
# data = featureFormat(data_dict, features)
#
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

### Remove 'TOTAL' (see 'Remove outliers' below)

data_dict.pop( 'TOTAL', 0 )
print "New length:", len(data_dict)

### Extract features and labels from dataset for local testing
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = [ 'poi', 'financial', 'salary', 'bonus'] # You will need to use more features
data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

print "NAN:", NAN_detector(features)

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# RFE (recursive feature elimination)

from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
clf = DecisionTreeClassifier(min_samples_split=40)
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
from sklearn.metrics import accuracy_score
print "DT accuracy:", accuracy_score(pred, labels_test)

selector = RFE(clf,3, step=1)
selector = selector.fit(features_train, labels_train)
print "RFE score:", selector.support_

### Define classifiers

# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
# from sklearn.svm import SVC
# clf = SVC()
# from sklearn import tree
# clf = tree.DecisionTreeClassifier()
#
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(
        n_neighbors=3,
        weights='distance',
        metric = 'minkowski')

clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print "KNN accuracy:", accuracy_score(pred, labels_test)

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
