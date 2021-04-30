""""

"""

import math
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import BaggingClassifier as BC
from sklearn.preprocessing import OneHotEncoder, RobustScaler, Normalizer

CATEGORICAL = [1, 2, 10, 11]
CONTINUOUS = [0, 3, 4, 5, 6, 7, 8, 9, 12]

def main():
    # Get the first 14 features of 297 samples
    patients = read_file('processed.cleveland.data', 297, range(14))
    # patients = read_file('processed_test.data', 282, range(14))

    # Binarize output column
    patients[:,13] = np.where(patients[:,13] > 0, 1, 0)

    # Shuffle input data
    np.random.shuffle(patients)

    # Show predetermined graphs
    # display(patients)

    # Hyperparameter tuning of KNN approach
    k_params = range(1,15)
    min_k = None
    min_error = 1
    roc = []
    # accuracy = []
    for k in k_params:
        kclf = KNN(n_neighbors=k)
        error, spec, sens, acc = trainAndTest(patients, kclf)
        error = np.mean(error)
        roc.append([spec, sens])
        if error < min_error:
            min_error = error
            min_k = k
    # display_ROC(np.array(roc), "ROC for KNN")
    kclf = KNN(n_neighbors=14) # check knn accuracy w 14 folds
    display_Accuracy(patients, kclf, "Sample Count vs Accuracy For KNN")
    print("KNN error: {} with k = {}".format(min_error, min_k))

    # Hyperparameter tuning of Decision Tree approach 
    max_depth_params = range(1, 14)
    min_max_depth = None
    min_error = 1
    roc = []
    for max_depth in max_depth_params:
        tclf = DTC(max_depth=max_depth)
        bclf = BC(tclf, max_samples=0.5, max_features=0.5)
        error, spec, sens, accuracy = trainAndTest(patients, bclf)
        error = np.mean(error)
        roc.append([spec, sens])
        if error < min_error:
            min_error = error
            min_max_depth = max_depth
    # display_ROC(np.array(roc), "ROC for Decision Tree")
    tclf = DTC(max_depth=14) # check tree accuracy w 14 max depth
    bclf = BC(tclf, max_samples=0.5, max_features=0.5)
    display_Accuracy(patients, bclf, "Sample Count vs Accuracy For Decision Tree")
    print("Tree Classifier error: {} with max_depth = {}".format(min_error, min_max_depth))

def trainAndTest(data, model):
    '''
        Trains on data using model and tests using 5-fold-cross-validation

        Parameters:
            data (n samples x d features numpy array)
            model (sklearn-object)
    '''

    # ROC metrics
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    # Preprocess data
    n, _ = np.shape(data)
    X = data[:, :13]
    y = data[:, 13]
    encoder = OneHotEncoder().fit(X[:, CATEGORICAL])

    folds = 5
    z = np.zeros(folds)
    for i in range(folds):
        a = math.ceil(i * n / folds)
        b = math.floor((i + 1) * n / folds)
        T = np.arange(a, b, dtype=int)
        S = np.setdiff1d(np.arange(0, n), T)

        # data preprocessing
        X_train = X[S, :]
        X_train = preprocess(X_train, encoder)
        y_train = y[S]

        model.fit(X_train, y_train)

        X_test = X[T, :]
        X_test = preprocess(X_test, encoder)
        y_expected = y[T]



        for t in range(len(X_test)):
            X_t = X_test[t, None]
            y_test = model.predict(X_t)
            if y_expected[t] != y_test:
                z[i] += 1
                if y_expected[t] == 1:
                    false_neg += 1
                else:
                    false_pos += 1
            else:
                if y_expected[t] == 1:
                    true_pos += 1
                else:
                    true_neg += 1
        z[i] = z[i] / len(T)

    sens = true_pos / (true_pos + false_neg)
    spec = true_neg / (true_neg + false_pos)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_neg + false_pos)

    return z, spec, sens, accuracy

def preprocess(X, encoder):
    X_cat = X[:, CATEGORICAL] # categorial variables
    X_con = X[:, CONTINUOUS] # continuous variables

    X_con = RobustScaler().fit_transform(X_con)
    X_con = Normalizer().fit_transform(X_con)
    X_cat = encoder.transform(X_cat).toarray()
    X = np.hstack((X_con, X_cat))
    return X

def display(data):
    '''
        Displays various charts and graphs to analyze data

        Parameters:
            data (n samples x d features numpy array)
    '''

    # # 3D Plots
    # X_0 = data[data[:, 13] == 0., :]
    # X_1 = data[data[:, 13] == 1., :]

    # sns.set(style='darkgrid')
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(X_0[:, 7], X_0[:, 0], X_0[:, 9], c='b')
    # ax.scatter(X_1[:, 7], X_1[:, 0], X_1[:, 9], c='r')

    # ax.set_xlabel('Maximum Heart Rate')
    # ax.set_ylabel('Age')
    # ax.set_zlabel('ST depression')

    # Relevant histograms
    fig, axs = plt.subplots()

    # create_hist(axs, data, 0, "Age")
    # create_hist(axs, data, 1, "Sex (1-Male, 0-Female)")
    # create_hist(axs, data, 9, "Rest Blood Pressure (mm Hg)")
    create_hist(axs, data, 7, "Maximum Heart Rate")

    # Show any plots created
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9,
                        top=0.9, wspace=0.4, hspace=0.4)

    plt.show()

def display_ROC(data, title):
    '''
        Displays ROC curve for various hyperparameters.

        Parameters:
            data (n samples x 2 (specificity and sensitivity))
    '''

    # Sort by specificity
    data = data[data[:, 0].argsort()]

    # Plot line chart
    fig, axs = plt.subplots()
    plt.scatter(1 - data[:, 0], data[:, 1])
    plt.title(title)
    plt.xlabel('1 - Specificity')
    # plt.xlim([0, 1])
    plt.ylabel('Sensitivity')
    # plt.ylim([0, 1])
    plt.show()

def display_Accuracy(patients, model, title):

    objects = ('30', '40', '50', '75', '100', '150')
    y_pos = np.arange(len(objects))
    accuracy = []

    # get avg accuracy value for 30, 40, 50, 75, 100, 150 samples across from subsets of dataset
    
    sample_sizes = [30, 40, 50, 75, 100, 150]

    for sample_size in sample_sizes:
        # get avg accuracy for 30 samples across subsets of dataset
        prev_ind = 0
        sample_avg = 0
        i = 0
        for t in range(sample_size - 1, 296, sample_size):
            i = i + 1
            error, spec, sens, acc = trainAndTest(patients[prev_ind:t,:], model)
            prev_ind = t
            sample_avg = (sample_avg + acc)

        accuracy.append(sample_avg / i)

    plt.bar(y_pos, accuracy, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Accuracy %')
    plt.xlabel('# of samples')
    plt.title(title)
    plt.show()


def create_hist(axs, data, feature, feature_label):
    colors = ['orange', 'blue']
    labels = ['With heart disease', 'Without heart disease']
    axs.hist(get_samples(data, 13, 0)[:, feature],
                density=True, color=colors[0], label=labels[0])
    axs.hist(get_samples(data, 13, 1)[:, feature],
                density=True, alpha=0.5, color=colors[1], label=labels[1])
    axs.legend()
    axs.set_xlabel(feature_label)
    axs.grid(True)

def get_samples(data, feature, val):
    '''
        Gets samples of data where feature matches value

        Parameters:
            data: original numpy array
            feature: index of feature to be checked
            val: expected value of feature
    '''
    return data[data[:, feature] == val, :]


def read_file(fileName, n, features):
    '''
        Reads data from a preprocessed file into a numpy array

        Parameters:
            fileName: CSV file containing samples by row and features by comma
            n: number of samples intended to read
            features: list of features to be used
                      (e.g. [0, 2, 4] would only get features 0, 2 and 4)

        Returns:
            data (n x len(features) numpy array): samples read from CSV file
    '''
    data = np.zeros((n, len(features)))
    with open(fileName, 'r', encoding='ISO-8859-1') as f:
        for i in range(n):
            try:
                # Convert each line into a numpy array of floats
                vals = np.array(f.readline().split(',')).astype(np.float64)
                data[i] = vals[features]
            except ValueError:
                # Discard line and warn of value error
                if f.readline() == '':
                    print("EOF reached, only read {} of {} samples".format(i, n))
                    data = np.delete(data, range(i, n), 0)
                    break
                else:
                    print("Warning: Unexpected value on line {}.".format(i))
    return data


def process_file(readFilename, writeFilename):
    # 303 patients with 76 data points
    # data after the 282nd patient seems to be corrupted
    patients_all_attr = np.zeros((282, 76))
    patients = np.zeros((282, 14))

    # file = open(fileName, "r", errors='ignore')
    readFile = open(readFilename, "r", encoding='ISO-8859-1')

    # read contents of file into a string
    contents = readFile.read()

    # split string by space and newlines
    split_contents = contents.replace("\n", " ").split()

    name_count = 0
    p_num = 0
    p_attr = 0
    for i in split_contents:
        if i == "name":
            p_num += 1
            p_attr = 0
            name_count += 1

            # the dataset seems to be corrupted after the 282nd patient
            if name_count > 281:
                break

        else:
            patients_all_attr[p_num][p_attr] = i
            p_attr += 1

    # process into 14 attributes
    i = 0
    for patient in patients_all_attr:
        # all n in patient[n] is subtracted by 1 because 0 indexing
        patients[i][0] = patient[2]  # age
        patients[i][1] = patient[3]  # sex
        patients[i][2] = patient[8]  # cp
        patients[i][3] = patient[9]  # trestbps
        patients[i][4] = patient[11]  # chos
        patients[i][5] = patient[15]  # fbs
        patients[i][6] = patient[18]  # restecg
        patients[i][7] = patient[31]  # thalach
        patients[i][8] = patient[37]  # exang
        patients[i][9] = patient[39]  # oldpeak
        patients[i][10] = patient[40]  # slope
        patients[i][11] = patient[43]  # ca
        patients[i][12] = patient[50]  # thal
        patients[i][13] = patient[57]  # num (predicted atttribute)
        i += 1

    # write into file
    writeFile = open(writeFilename, "w")
    for patient in patients:
        for iter in range(np.size(patient)):
            writeFile.write(str(patient[iter]))

            if iter < np.size(patient) - 1:
                writeFile.write(",")

        writeFile.write("\n")

    readFile.close()
    writeFile.close()


if __name__ == "__main__":
    main()
