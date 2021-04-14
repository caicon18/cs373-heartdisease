import numpy as np


def main():
    # file = open("cleveland.data", "r")  # this creates a decoding error

    # 303 patients with 76 data points
    # data after the 282nd patient seems to be corrupted
    patients = np.zeros((282, 76))

    # file = open("cleveland.data", "r", errors='ignore')
    file = open("cleveland.data", "r", encoding='ISO-8859-1')

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
            patients[p_num][p_attr] = i
            p_attr += 1

    print(str(patients))

    file.close()
    return 0


main()
