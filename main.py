import numpy as np

def main():
    patients = process_file("cleveland.data")

    print(patients)
    return 0


def process_file(fileName):
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


main()
