from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
import numpy as np

# script for preprocessing the mnist data and saving it to disk

mnist = fetch_openml('mnist_784')
X = mnist['data'].astype('float64').to_numpy()
t = mnist['target'].astype('int').to_numpy()
random_state = check_random_state(1)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
t = t[permutation]
# This line flattens the image into a vector of size 784
X = X.reshape((X.shape[0], -1))

# Add a column of ones for the bias term
X = np.hstack((X, np.ones((X.shape[0], 1))))

# Split the data into train, validation and test sets, with 60%, 20%, 20% of the data respectively
X_train_val, X_test, t_train_val, t_test = train_test_split(
    X, t, test_size=0.2, random_state=1)
X_train, X_val, t_train, t_val = train_test_split(
    X_train_val, t_train_val, test_size=0.25, random_state=1)

# The next lines standardize the images
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Save the preprocessed data to disk
np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)
np.save('X_test.npy', X_test)
np.save('t_train.npy', t_train)
np.save('t_val.npy', t_val)
np.save('t_test.npy', t_test)
