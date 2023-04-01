import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier as RFC

#TODO plot b on the same axis: have a plot function
#TODO ask professor if we should do cv on any hyperparameter for lda and random forest or just work with default params
#TODO learning the best n_estimators for random forest would be great
#TODO find out if average precision is the mean of precision (positive) and precision (negative)
#TODO check calculation for precision, f1 score and recall
#TODO separate c and d


def load_and_split_data(filename):
    """
    loads the dataset, and separates the data and the label
    :param filename: the full path of the input file
    :return: the data and the label in the file
    """
    dataset = []
    data_str = []
    label_str = []
    # Read the data line by line and append to a list
    with open(filename, 'r') as file:
        for line in file:
            dataset.append(line.strip())

    # separate the data and label
    for i in range(len(dataset)):
        data_str.append(dataset[i][:-2])
        label_str.append(dataset[i][-1])

    # convert both data and label into a numpy array
    new_label = [string.replace("B", "0").replace("M", "1") for string in label_str]
    label = np.array(new_label).astype(np.float64)

    new_data = [[num for num in string.split(',')] for string in data_str]
    data = np.array(new_data).astype(np.float64)
    return data, label


def configure_svm(x_train, y_train, Cs, cv, scoring):
    scores = []
    for C in Cs:
        classifier = SVC(kernel='linear', C=C, random_state=42)
        scores_cv = cross_val_score(classifier, x_train, y_train, cv=cv, scoring=scoring)
        scores.append(np.mean(scores_cv))

    print(f'C: {Cs}')
    print(f'scores: {scores}')
    plt.figure(1)
    plt.plot(Cs, scores, '-*')
    plt.xlabel("C")
    plt.ylabel("mean F1 scores")
    plt.title("plot of scores vs C")
    plt.show()

    max_index = scores.index(max(scores))
    return Cs[max_index]


def configure_decision_tree(x_train, y_train, ks, cv, scoring, criterion, figure):
    scores = []
    for k in ks:
        classifier = DecisionTreeClassifier(criterion=criterion, random_state=0, max_leaf_nodes=k)
        scores_cv = cross_val_score(classifier, x_train, y_train, cv=cv, scoring=scoring)
        scores.append(np.mean(scores_cv))

    print(f'k: {ks}')
    print(f'scores: {scores}')
    plt.figure(figure)
    plt.plot(ks, scores, '-*')
    plt.xlabel("k")
    plt.ylabel("mean F1 scores")
    plt.title(f"plot of scores vs k for DT-{criterion}")
    plt.show()

    max_index = scores.index(max(scores))
    return ks[max_index]


def grid_search(x_train, y_train, Cs, cv):
    classifier = SVC(kernel='linear', C=Cs, random_state=42)
    from sklearn.model_selection import GridSearchCV
    parameters = [{'C': [0.01, 0.1, 1, 10, 100]}]
    grid_searchs = GridSearchCV(estimator=classifier,
                                param_grid=parameters,
                                scoring='f1',
                                cv=cv,
                                n_jobs=-1)
    grid_searchs.fit(x_train, y_train)
    best_accuracy = grid_searchs.best_score_
    best_param = grid_searchs.best_params_
    print(f'Best C: {best_param}')
    print(f'best accuracy: {best_accuracy}')


def calculate_evaluation_metrics(tn, fp, fn, tp):
    """
    shows the calculation of recall, precision, accuracy, fscore, etc.
    :return: none
    """
    accuracy = (tn + tp) / (tn + fp + fn + tp)
    error_rate = (fn + fp) / (tn + fp + fn + tp)
    precision_p = tp / (tp + fp)
    precision_n = tn / (tn + fn)
    recall_p = tp / (tp + fn)
    recall_n = tn / (fp + tn)
    f1_p = (2 * precision_p * recall_p) / (precision_p + recall_p)
    f1_n = (2 * precision_n * recall_n) / (precision_n + recall_n)

    print(f'accuracy: {accuracy}')
    print(f'error rate: {error_rate}')
    print(f'precision (p): {precision_p}')
    print(f'precision (n): {precision_n}')
    print(f'recall (p): {recall_p}')
    print(f'recall (n): {recall_n}')
    print(f'f-measure (p): {f1_p}')
    print(f'f-measure (n): {f1_n}')

    precision = np.mean([precision_n, precision_p])
    recall = np.mean([recall_n, recall_p])
    f1 = np.mean([f1_n, f1_p])

    return precision, recall, f1


def plot_bar(data, label, eval_metrics):
    plt.bar(label, data)
    plt.title("Bar Graph")
    plt.xlabel("label")
    plt.ylabel(f"{eval_metrics}")
    plt.show()


def main():
    train_data = "./cancer-data-train.csv"
    test_data = "./cancer-data-test.csv"
    C = [0.01, 0.1, 1, 10, 100]
    k = [2, 5, 10, 20]
    cv = 10
    scoring = 'f1'
    criterion = ['gini', 'entropy']

    x_train, y_train = load_and_split_data(train_data)
    x_test, y_test = load_and_split_data(test_data)
    best_c = configure_svm(x_train, y_train, C, cv, scoring)
    # grid_search(x_train, y_train, C, cv)
    best_k_DT_gini = configure_decision_tree(x_train, y_train, k, cv, scoring, criterion[0], 2)
    best_k_DT_ig = configure_decision_tree(x_train, y_train, k, cv, scoring, criterion[1], 3)

    # train and test SVM using best C learned above
    svm_classifier = SVC(kernel='linear', C=best_c, random_state=42)
    svm_classifier.fit(x_train, y_train)
    y_pred_svm = svm_classifier.predict(x_test)
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred_svm).ravel()
    print(tn, fp, fn, tp)

    # train and test DT-gini using the best k (max leaf nodes) learned above
    dt_gini_classifier = DecisionTreeClassifier(criterion=criterion[0], random_state=0, max_leaf_nodes=best_k_DT_gini)
    dt_gini_classifier.fit(x_train, y_train)
    y_pred_gini = dt_gini_classifier.predict(x_test)
    tn_gini, fp_gini, fn_gini, tp_gini = metrics.confusion_matrix(y_test, y_pred_gini).ravel()
    print(tn_gini, fp_gini, fn_gini, tp_gini)

    # train and test DT-ig using the best k (max leaf nodes) learned above
    dt_ig_classifier = DecisionTreeClassifier(criterion=criterion[1], random_state=0, max_leaf_nodes=best_k_DT_ig)
    dt_ig_classifier.fit(x_train, y_train)
    y_pred_ig = dt_ig_classifier.predict(x_test)
    tn_ig, fp_ig, fn_ig, tp_ig = metrics.confusion_matrix(y_test, y_pred_ig).ravel()
    print(tn_ig, fp_ig, fn_ig, tp_ig)

    # train and test LDA using default hyper-parameters
    lda_classifier = LDA()                  # LDA - Linear Discriminant Analysis see import
    lda_classifier.fit(x_train, y_train)
    y_pred_lda = lda_classifier.predict(x_test)
    tn_lda, fp_lda, fn_lda, tp_lda = metrics.confusion_matrix(y_test, y_pred_lda).ravel()
    print(tn_lda, fp_lda, fn_lda, tp_lda)

    # train and test random forest using default hyper-parameters
    rf_classifier = RFC(n_estimators=10, random_state=0)                   # RFC - Random forest classifier see import
    rf_classifier.fit(x_train, y_train)
    y_pred_rfc = rf_classifier.predict(x_test)
    tn_rfc, fp_rfc, fn_rfc, tp_rfc = metrics.confusion_matrix(y_test, y_pred_rfc).ravel()
    print(tn_rfc, fp_rfc, fn_rfc, tp_rfc)

    # calculate evaluation metrics
    precision_svm, recall_svm, f1_svm = calculate_evaluation_metrics(tn, fp, fn, tp)
    precision_gini, recall_gini, f1_gini = calculate_evaluation_metrics(tn_gini, fp_gini, fn_gini, tp_gini)
    precision_ig, recall_ig, f1_ig = calculate_evaluation_metrics(tn_ig, fp_ig, fn_ig, tp_ig)
    precision_lda, recall_lda, f1_lda = calculate_evaluation_metrics(tn_lda, fp_lda, fn_lda, tp_lda)
    precision_rfc, recall_rfc, f1_rfc = calculate_evaluation_metrics(tn_rfc, fp_rfc, fn_rfc, tp_rfc)

    # plot bar graphs
    label = ["SVM", "DT-GINI", "DT-IG", "LDA", "RF"]
    precision_data = [precision_svm, precision_gini, precision_ig, precision_lda, precision_rfc]
    recall_data = [recall_svm, recall_gini, recall_ig, recall_lda, recall_rfc]
    f1_data = [f1_svm, f1_gini, f1_ig, f1_lda, f1_rfc]

    plot_bar(precision_data, label, "Precision")
    plot_bar(recall_data, label, "Recall")
    plot_bar(f1_data, label, "f1 score")



if __name__ == '__main__':
    main()
