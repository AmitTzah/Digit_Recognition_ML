from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
import numpy as np

mnist = fetch_openml('mnist_784')
X = mnist['data'].astype('float64').to_numpy()
t = mnist['target'].astype('int').to_numpy()
random_state = check_random_state(1)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
t = t[permutation]
X = X.reshape((X.shape[0], -1)) #This line flattens the image into a vector of size 784
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.2)
# The next lines standardize the images
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the preprocessed data to disk
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('t_train.npy', t_train)
np.save('t_test.npy', t_test)