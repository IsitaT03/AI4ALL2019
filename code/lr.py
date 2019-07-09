## Logistic regression
from sklearn.linear_model import LogisticRegression
from lib import *
from classifiers import *

def my_logisticregression(data_store):
    train_set = data_store.get_train_set()
    test_set = data_store.get_test_set()
    train_profile = [x.get_gene_profile() for x in train_set]
    train_labels = [x.get_label() for x in train_set]

    clf = LogisticRegression().fit(train_profile, train_labels)
    #now we predict
    test_profile = [x.get_gene_profile() for x in test_set]
    test_labels = [x.get_gene_profile() for x in test_set]
    pred = clf.predict(test_profile)
    print("default...")
    evaluate_results(list(zip(test_set, pred)))

    #now we put in some penality predict
    clf = LogisticRegression(penalty="l1").fit(train_profile, train_labels)
    pred = clf.predict(test_profile)
    print("using L1...")
    evaluate_results(list(zip(test_set, pred)))
