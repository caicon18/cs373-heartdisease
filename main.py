""""

"""

import math
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from sklearn.metrics import auc, plot_roc_curve
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

    # ROC results
    rocs = []

    # Hyperparameter tuning of KNN approach
    k_params = range(1,20)
    min_k = None
    min_error = 1
    tprs = []
    aucs = []

    for k in k_params:
        kclf = KNN(n_neighbors=k)
        error, k_tprs, k_aucs, _  = trainAndTest(patients, kclf)
        error = np.mean(error)
        tprs = tprs + k_tprs
        aucs = aucs + k_aucs

        if error < min_error:
            min_error = error
            min_k = k

    rocs.append([tprs, aucs, (1, 0, 0), 'KNN'])
    print("KNN error: {} with k = {}".format(min_error, min_k))

    # Hyperparameter tuning of Decision Tree approach 
    max_depth_params = range(1, 14)
    min_max_depth = None
    min_error = 1
    tprs = []
    aucs = []

    for max_depth in max_depth_params:
        tclf = DTC(max_depth=max_depth)
        bclf = BC(tclf, max_samples=0.5, max_features=0.5)
        error, tprs, aucs, _ = trainAndTest(patients, bclf)
        error = np.mean(error)
        tprs = tprs + k_tprs
        aucs = aucs + k_aucs

        if error < min_error:
            min_error = error
            min_max_depth = max_depth

    rocs.append([tprs, aucs, (0, 0, 1), 'Tree Classifier'])
    print("Tree Classifier error: {} with max_depth = {}".format(min_error, min_max_depth))

    # Display results
    kclf = KNN(n_neighbors=14) # check knn accuracy w 14 folds
    display_Accuracy(patients, kclf, "Sample Count vs Accuracy For KNN")
    tclf = DTC(max_depth=14) # check tree accuracy w 14 max depth
    bclf = BC(tclf, max_samples=0.5, max_features=0.5)
    display_Accuracy(patients, bclf, "Sample Count vs Accuracy For Decision Tree")
    display_ROC(rocs, "ROC Plot")

def trainAndTest(data, model):
    '''
        Trains on data using model and tests using 5-fold-cross-validation

        Parameters:
            data (n samples x d features numpy array)
            model (sklearn-object)
    '''

    # Result metrics
    tprs = []
    aucs = []
    accs = []
    mean_fpr = np.linspace(0, 1, 100)

    # Transform data and fit number of categories
    n, _ = np.shape(data)
    X = data[:, :13]
    y = data[:, 13]
    encoder = OneHotEncoder().fit(X[:, CATEGORICAL])

    # Perform 5-fold cross-validation
    folds = 5
    z = np.zeros(folds)
    for i in range(folds):
        a = math.ceil(i * n / folds)
        b = math.floor((i + 1) * n / folds)
        T = np.arange(a, b, dtype=int)
        S = np.setdiff1d(np.arange(0, n), T)

        # Preprocess training data
        X_train = X[S, :]
        X_train = preprocess(X_train, encoder)
        y_train = y[S]

        # Train on taining data
        model.fit(X_train, y_train)

        # Preprocess testing data
        X_test = X[T, :]
        X_test = preprocess(X_test, encoder)
        y_test = y[T]
    
        # Predict and compare on testing data
        y_pred = model.predict(X_test)
        z[i] = np.sum(y_pred != y_test) / len(T)

        false_pos = np.sum(np.logical_and(y_pred == 1, y_test == 0))
        false_neg = np.sum(np.logical_and(y_pred == 0, y_test == 1))
        true_pos = np.sum(np.logical_and(y_pred == 1, y_test == 1))
        true_neg = np.sum(np.logical_and(y_pred == 0, y_test == 0))
        
        sum_a = true_pos + false_neg
        sum_b = true_neg + false_pos

        if sum_a != 0:
            tpr = [0., true_pos / (sum_a), 1.]
        else:
            tpr = [0., 1.]
        
        if sum_b != 0:
            fpr = [0., 1 - (true_neg / (sum_b)), 1.]
        else:
            fpr = [0., 1.]
        
        acc = (true_pos + true_neg) / (sum_a + sum_b)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(np.trapz(tpr, fpr))
        accs.append(acc)

    return z, tprs, aucs, np.mean(accs)

def preprocess(X, encoder):
    X_cat = X[:, CATEGORICAL] # categorial variables
    X_con = X[:, CONTINUOUS] # continuous variables

    X_con = RobustScaler().fit_transform(X_con)
    X_con = Normalizer().fit_transform(X_con)
    X_cat = encoder.transform(X_cat).toarray()
    X = np.hstack((X_con, X_cat))
    return X

def display_ROC(rocs, title):
    '''
        Displays ROC curve for various hyperparameters.

        Parameters:
            roc (n classifiers x 4 (true positivity rates,
                                    area under curves,
                                    color and name))
    '''
    mean_fpr = np.linspace(0, 1, 100)
    for roc in rocs:
        tprs = roc[0]
        aucs = roc[1]
        color = roc[2]
        name = roc[3]

        # Plot interpolated ROC line
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color=color,
                label=r'Mean ROC for %s (AUC = %0.2f $\pm$ %0.2f)' %
                     (name, mean_auc, std_auc),
                lw=2, alpha=.8)

        # Plot area of std deviation
        dev_color = (color[0], color[1], color[2], 0.25)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper,
                            color=dev_color, alpha=.2,
                            label=r'%s$\pm$ 1 std. dev.' %
                                    (name))

    # Plot line of random classifier for ref
    plt.plot([-.15, 1.15], [-.15, 1.15], color='black', lw=2)

    plt.title(title)
    plt.ylabel("Sensitivity")
    plt.ylim([0, 1])
    plt.xlabel("1 - Specification")
    plt.xlim([0, 1])
    plt.legend(loc="lower right")
    plt.show()

def display_Accuracy(patients, model, title):

    objects = ('30', '40', '50', '75', '100', '150')
    y_pos = np.arange(len(objects))
    accuracy = []

    # get avg accuracy value for 30, 40, 50, 75, 100, 150 samples
    # across from subsets of dataset
    
    sample_sizes = [30, 40, 50, 75, 100, 150]

    for sample_size in sample_sizes:
        # get avg accuracy for 30 samples across subsets of dataset
        prev_ind = 0
        sample_avg = 0
        i = 0
        for t in range(sample_size - 1, 296, sample_size):
            i = i + 1
            error, _, _, acc = trainAndTest(patients[prev_ind:t,:], model)
            prev_ind = t
            sample_avg = (sample_avg + acc)

        accuracy.append(sample_avg / i)

    plt.bar(y_pos, accuracy, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Accuracy %')
    plt.xlabel('# of samples')
    plt.title(title)
    plt.show()

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
