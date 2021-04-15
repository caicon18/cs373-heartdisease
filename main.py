import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def main():
    # patients = process_file("cleveland.data")

    # Get the first 14 features of 297 samples
    patients = read_file('processed.cleveland.data', 297, range(14))

    # Relevant histograms (uncomment intended distribution to analyze)
    features = range(2,4)
    fig, axs = plt.subplots(len(features), 1)

    for i in range(len(axs)):
        # Feature without heart disease
        colors = ['orange', 'blue']
        labels = ['With heart disease', 'Without heart disease']
        axs[i].hist(get_samples(patients, 13, 0)[:, features[i]],
                    density=True, color=colors[0], label=labels[0])
        axs[i].hist(get_samples(patients, 13, 1)[:, features[i]],
                    density=True, alpha=0.5, color=colors[1], label=labels[1])
        axs[i].legend()
        axs[i].set_xlabel('Value Feature {}'.format(features[i]))
        axs[i].yaxis.set_major_formatter(PercentFormatter(xmax=1))
        axs[i].grid(True)

    # Show any plots created
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9,
                        top=0.9, wspace=0.4, hspace=0.4)
    plt.show()

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
                vals = np.array(f.readline().split(',')).astype(np.float)
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
            

def process_file(fileName):
    # TODO: Modify function to write to processed file instead of returning data
    # 303 patients with 76 data points
    # data after the 282nd patient seems to be corrupted
    patients_all_attr = np.zeros((282, 76))
    patients = np.zeros((282, 14))

    # file = open(fileName, "r", errors='ignore')
    file = open(fileName, "r", encoding='ISO-8859-1')

    # read contents of file into a string
    contents = file.read()

    # split string by space and newlines
    contents = contents.replace("\n", " ")
    print(contents)
    print()
    split_contents = contents.split()  # this seems to sprinkle \x00 values in the array

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

    file.close()
    return patients


if __name__ == "__main__":
    main()