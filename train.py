import numpy as np

# Using multinomial logistic regression to classify the MNIST dataset


def load_data():

    # Check if preprocessed data exists
    try:
        X_train = np.load('X_train.npy')
        X_test = np.load('X_test.npy')
        t_train = np.load('t_train.npy')
        t_test = np.load('t_test.npy')
        t_val = np.load('t_val.npy')
        X_val = np.load('X_val.npy')

        print("Preprocessed data found. Loading from files.")

        return X_train, X_test, t_train, t_test, X_val, t_val

    except FileNotFoundError:
        print("Preprocessed data not found, run mnist.py first.")


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


def cross_entropy_loss(W, X, t):
    dot_products_matrix = np.dot(X, W.T)
    exp_dot_products_matrix = np.exp(dot_products_matrix)

    # divide each element of the matrix by the sum of the elements in the row
    # this way, softmax_matrix[n][k] will be the probability of sample n being in class k
    # For testing purposes, we can check that softmax_matrix[n][k] is equal to softmax(dot_products_matrix, n, k)
    softmax_matrix = exp_dot_products_matrix / \
        np.sum(exp_dot_products_matrix, axis=1, keepdims=True)

    # We use matrix multiplications because it is a lot faster than using double for loops
    loss = -np.sum(t * np.log(softmax_matrix))
    return loss


def init_weights():

    # first check if we have a best_weights.npy file already set up with good weights
    try:
        W = np.load('best_weights.npy')
        print("Best weights found. Loading from file.")
        return W
    except FileNotFoundError:
        print("Best weights not found. Initializing random weights.")

    # Initialize 10 random vectors W_0, W_1, ..., W_9 with a length of 785
    # these are the weights for each of the 10 classes
    num_vectors = 10
    vector_length = 785

    # initialize a matrix of size 10x785 with random values
    # To avoid possible overflow, we initialize the weights to be small

    W = np.random.rand(num_vectors, vector_length) * 0.0001

    return W


def gradient_descent(W, X, t, learning_rate, num_iterations):
    iterations = 0
    num_samples = X.shape[0]

    while iterations < num_iterations:
        dot_products_matrix = np.dot(X, W.T)
        exp_dot_products_matrix = np.exp(dot_products_matrix)

        softmax_matrix = exp_dot_products_matrix / \
            np.sum(exp_dot_products_matrix, axis=1, keepdims=True)

        sample_error = softmax_matrix - t

        # Calculate the weight update step
        grad = np.dot(sample_error.T, X)
        W = W - learning_rate * grad

        # Print the loss every 10 iterations
        if iterations % 10 == 0:
            print("Iteration:", iterations, "Loss:",
                  cross_entropy_loss(W, X, t))

        iterations += 1

    return W


def main():

    # Load the preprocessed data
    data = load_data()

    if data is not None:

        X_train, X_test, t_train, t_test, X_val, t_val = data

    else:
        return

    # Initialize the weights
    W = init_weights()

    # Set the learning rate
    learning_rate = 0.00005

    # Set the number of iterations
    num_iterations = 300

    # Train the model
    W = gradient_descent(W, X_train, t_train, learning_rate, num_iterations)

    # save the weights to disk
    np.save('best_weights.npy', W)


if __name__ == '__main__':
    main()
