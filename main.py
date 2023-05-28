import matplotlib.pyplot as plt
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


def calculate_accuracy(W, X, t):

    # get the softmax matrix
    dot_products_matrix = np.dot(X, W.T)
    exp_dot_products_matrix = np.exp(dot_products_matrix)

    softmax_matrix = exp_dot_products_matrix / \
        np.sum(exp_dot_products_matrix, axis=1, keepdims=True)

    # The predicted class is the class with the highest probability
    # For each row, find the index of the highest probability
    predicted_classes = np.argmax(softmax_matrix, axis=1)

    # The true class is the class with the value 1 in the t vector
    # We can find the true class by finding the index of the value 1 in each row
    true_classes = np.argmax(t, axis=1)

    # The accuracy is the percentage of correctly classified samples
    accuracy = np.sum(predicted_classes == true_classes) / len(true_classes)

    return accuracy


def plot_loss_and_accuracy(train_losses, val_accuracies):

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(len(val_accuracies)), val_accuracies,
             label='Validation Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save the plot as a PNG file
    plt.savefig('plot.png')

    # Close the plot to free resources
    plt.close()


def gradient_descent(W, X_train, t_train, X_val, t_val, learning_rate, num_iterations, early_stopping_patience):
    iterations = 0
    best_val_accuracy = 0
    no_improvement_counter = 0

    train_losses = []
    val_accuracies = []

    while iterations < num_iterations:
        dot_products_matrix = np.dot(X_train, W.T)
        exp_dot_products_matrix = np.exp(dot_products_matrix)

        softmax_matrix = exp_dot_products_matrix / \
            np.sum(exp_dot_products_matrix, axis=1, keepdims=True)

        sample_error = softmax_matrix - t_train

        # Calculate losses and accuracy
        train_loss = cross_entropy_loss(W, X_train, t_train)
        val_loss = cross_entropy_loss(W, X_val, t_val)
        val_accuracy = calculate_accuracy(W, X_val, t_val)

        # Append the values for plotting
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)

        # Calculate the gradient and update the weights
        grad = np.dot(sample_error.T, X_train)
        W = W - learning_rate * grad

        # Print the loss and accuracy every 10 iterations
        if iterations % 10 == 0:
            print("Iteration:", iterations, "Train Loss:", train_loss,
                  "Val Loss:", val_loss, "Val Accuracy:", val_accuracy)

            # Check for early stopping
            # If the validation accuracy is better by 5% than the best validation accuracy so far, reset the counter
            if val_accuracy > 1.05 * best_val_accuracy:
                best_val_accuracy = val_accuracy
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1
                if no_improvement_counter >= early_stopping_patience:
                    print("validation accuracy did not improve by 5% for", 10 *
                          early_stopping_patience, "iterations. Stopping early.")
                    break

        iterations += 1

    # Plot the results

    plot_loss_and_accuracy(train_losses, val_accuracies)

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

    # check accuracy before training
    print("Accuracy before training:", calculate_accuracy(W, X_train, t_train))

    # Set the learning rate
    learning_rate = 0.000001

    # Set the number of iterations
    num_iterations = 100

    # Train the model
    W = gradient_descent(W, X_train, t_train, X_val, t_val,
                         learning_rate, num_iterations, early_stopping_patience=10)

    print("finished training, saving best weights to file for later use")

    # save the weights to disk
    np.save('best_weights.npy', W)

    # Calculate the accuracy on the test set, valid set and train set
    print("Final accuracies:")
    print("Train accuracy:", calculate_accuracy(W, X_train, t_train))
    print("Validation accuracy:", calculate_accuracy(W, X_val, t_val))
    print("Test accuracy:", calculate_accuracy(W, X_test, t_test))


if __name__ == '__main__':
    main()
