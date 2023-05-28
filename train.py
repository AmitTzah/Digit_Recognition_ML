import numpy as np

#Using multinomial logistic regression to classify the MNIST dataset

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


# compute y_n_k, the softmax function, the probability of each class
# n is the index of the sample, k is the index of the class
def softmax(dot_products_matrix, n, k):
    # W is a matrix of size 10x785 (10 classes and 785 weights for each class)
    # X is a matrix of size NumberOfSamplesx785 (number of samples and 785 features)

    # dot_products_matrix the dot product of X and W.T
    # this will give us a matrix of size NumberOfSamplesx10
    # each row of the matrix is the dot product of a sample and the 10 weight vectors

    exp_dot_products_matrix = np.exp(dot_products_matrix)

    numerator = exp_dot_products_matrix[n][k]

    denominator = np.sum(exp_dot_products_matrix[n])

    return numerator / denominator
  
    


# define the cross entropy loss function
def cross_entropy_loss(W, X, t):
    # W is a matrix of size 10x785
    # X is a matrix of size NumberOfSamplesx785
    # t is a matrix of size NumberOfSamplesx10
    # lam is a scalar value
    
    #get the number of samples
    NumberOfSamples = X.shape[0]

    #for computing the softmax function
    dot_products_matrix = np.dot(X, W.T)


    #sum over all the samples and all the classes
    sum = 0
    for n in range(NumberOfSamples):
        for k in range(10):
            sum += t[n][k] * np.log(softmax(dot_products_matrix, n, k))
        
    return -sum 
   


def init_weights():
    # Initialize 10 random vectors W_0, W_1, ..., W_9 with a length of 785
    # these are the weights for each of the 10 classes
    num_vectors = 10
    vector_length = 785

    W = np.random.randn(num_vectors, vector_length)

    return W


def main():

    # Load the preprocessed data
    data = load_data()

    if data is not None:

        X_train, X_test, t_train, t_test = data

    # Initialize the weights
    W = init_weights()

    cross_entropy_loss(W, X_train, t_train)


if __name__ == '__main__':
    main()
