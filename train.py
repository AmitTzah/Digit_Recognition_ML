import numpy as np


def load_data():

    # Check if preprocessed data exists
    try:
        X_train = np.load('X_train.npy')
        X_test = np.load('X_test.npy')
        t_train = np.load('t_train.npy')
        t_test = np.load('t_test.npy')
        print("Preprocessed data found. Loading from files.")

        return X_train, X_test, t_train, t_test

    except FileNotFoundError:
        print("Preprocessed data not found. run mnist.py first.")


def main():

    # Load the preprocessed data
    data = load_data()

    if data is not None:

        X_train, X_test, t_train, t_test = data


if __name__ == '__main__':
    main()
