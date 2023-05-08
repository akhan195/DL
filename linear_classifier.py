import numpy as np
import urllib.request
import gzip
import time
import matplotlib.pyplot as plt

# Download and load the data
def download_data():
    urllib.request.urlretrieve("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "train-images.gz")
    urllib.request.urlretrieve("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "train-labels.gz")
    
    # Download test data
    urllib.request.urlretrieve("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "test-images.gz")
    urllib.request.urlretrieve("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "test-labels.gz")

def load_data():
    with gzip.open('train-images.gz', 'rb') as f:
        train_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784).astype(np.float32) / 255.0

    with gzip.open('train-labels.gz', 'rb') as f:
        train_labels = np.frombuffer(f.read(), np.uint8, offset=8)

    with gzip.open('test-images.gz', 'rb') as f:
        test_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784).astype(np.float32) / 255.0

    with gzip.open('test-labels.gz', 'rb') as f:
        test_labels = np.frombuffer(f.read(), np.uint8, offset=8)

    

    return train_images, train_labels, test_images, test_labels

def normalize(x):
    # Convert inputs x to vectors of size 282 = 784.
    x = x.reshape(x.shape[0], -1)
    # Remove features with zero standard deviation
    std = np.std(x, axis=0)
    std[std == 0] = 1.0
    # Standardize input images using z-normalization
    mean = np.mean(x, axis=0)
    x = (x - mean) / std
    # Add bias variable by concatenating 1 to your input
    x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
    return x



def convert_labels(y):
    y_binary = np.zeros_like(y)
    y_binary[y > 4] = 1
    return y_binary

def linear_classifier(train_images, train_labels, test_images, test_labels, learning_rate):
    # Normalize input data
    train_images = normalize(train_images)
    test_images = normalize(test_images)
    # Convert labels to binary classification problem
    train_labels_binary = convert_labels(train_labels)
    test_labels_binary = convert_labels(test_labels)

    # Initialize weight matrix v using Xavier initialization
    k = train_images.shape[1]  # number of features
    v = np.random.randn(k + 1) / np.sqrt(k)

    # Initialize loss and accuracy lists for plotting
    train_loss_list = []
    test_loss_list = []
    train_accuracy_list = []
    test_accuracy_list = []

    # Train for 10 epochs
    for epoch in range(10):
        # Shuffle training data for each epoch
        indices = np.random.permutation(train_images.shape[0])
        train_images_shuffled = train_images[indices]
        train_labels_binary_shuffled = train_labels_binary[indices]

        # Iterate over mini-batches
        for i in range(0, train_images_shuffled.shape[0], 10):
            # Get mini-batch
            batch_images = train_images_shuffled[i:i+10]
            batch_labels = train_labels_binary_shuffled[i:i+10]

            # Forward pass
            y = np.dot(batch_images, v[:-1]) + v[-1]
            loss = np.mean((y - batch_labels)**2)

            # Backward pass
            gradient = np.mean(2 * (y - batch_labels)[:, np.newaxis] * batch_images, axis=0)
            gradient_bias = np.mean(2 * (y - batch_labels))
            v[:-1] -= learning_rate * gradient
            v[-1] -= learning_rate * gradient_bias

        # Compute and store training and test loss and accuracy
        train_loss = np.mean((np.dot(train_images, v[:-1]) + v[-1] - train_labels_binary)**2)
        test_loss = np.mean((np.dot(test_images, v[:-1]) + v[-1] - test_labels_binary)**2)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

        train_accuracy = np.mean((np.dot(train_images, v[:-1]) + v[-1] > 0.5) == train_labels_binary)
        test_accuracy = np.mean((np.dot(test_images, v[:-1]) + v[-1] > 0.5) == test_labels_binary)
        train_accuracy_list.append(train_accuracy)
        test_accuracy_list.append(test_accuracy)
        
        # Print test accuracy for each iteration
        print(f"Epoch {epoch+1}, Test Accuracy: {test_accuracy:.4f}")

    # Plot loss and accuracy curves
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(test_loss_list, label='Test Loss')
    plt.legend()
    plt.title('Quadratic Loss')
    plt.show()

    plt.plot(train_accuracy_list, label='Train Accuracy')
    plt.plot(test_accuracy_list, label='Test Accuracy')
    plt.legend()
    plt.show()

    # Return final test accuracy
    return test_accuracy



# Download and load the data
download_data()
train_images, train_labels, test_images, test_labels = load_data()

# Train the linear classifier
test_accuracy = linear_classifier(train_images, train_labels, test_images, test_labels, 0.0001)


print(f"Linear Classifier Test Accuracy: {test_accuracy:.4f}")