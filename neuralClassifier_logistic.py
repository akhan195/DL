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


# Convert labels to binary
def convert_labels(y):
    y_binary = np.zeros_like(y)
    y_binary[y > 4] = 1
    return y_binary


def logistic_classifier(train_images, train_labels, test_images, test_labels, k):
    W = np.random.normal(0, 1/np.sqrt(train_images.shape[1]), (k, train_images.shape[1]))
    v = np.random.normal(0, 1/np.sqrt(k), (k, len(np.unique(train_labels))))
    # Define the training parameters
    learning_rate = 0.1
    num_epochs = 10
    batch_size = 10
    num_batches = int(np.ceil(len(train_images) / batch_size))
    epsilon = 1e-10
    # Initialize the accuracy lists
    train_accuracy_his = []
    test_accuracy_his = []

    start_time = time.time()

    # Train the neural network
    for epoch in range(num_epochs):
    	#Shuffle training data for each epoch
    	indices = np.random.permutation(train_images.shape[0])
    	train_images_shuffled = train_images[indices]
    	train_labels_binary_shuffled = train_labels_binary[indices]
        print("EPoch num :", epoch)
        for i in range(0, train_images_shuffled.shape[0], 10):

            # Get the batch of data
            batch_data = train_images_shuffled[i:i+10]
      		batch_labels = train_labels_binary_shuffled[i:i+10]

            # Convert the batch labels to one-hot encoded matrix
            one_hot_labels = np.eye(len(np.unique(train_labels)))[batch_labels]

            # Forward pass
            hidden_activations = np.maximum(0, np.dot(batch_data, W.T))
            logits = np.dot(hidden_activations, v)
            predictions = 1 / (1 + np.exp(-logits))
            loss = -np.mean(one_hot_labels*np.log(predictions+epsilon) + (1-one_hot_labels)*np.log(1-predictions+epsilon))

            # Backward pass
            output_gradient = (predictions - one_hot_labels) / batch_size
            hidden_gradient = np.dot(output_gradient, v.T) * (hidden_activations > 0)
            v_gradient = np.dot(hidden_activations.T, output_gradient)
            W_gradient = np.dot(hidden_gradient.T, batch_data)

            # Update the weights
            v = v - learning_rate * v_gradient
            W = W - learning_rate * W_gradient

            # Compute the training and test accuracy
            train_logits = np.dot(np.maximum(0, np.dot(train_images, W.T)), v)
            train_predictions = np.argmax(train_logits, axis=1)
            train_accuracy = np.mean(train_predictions == train_labels)
            train_accuracy_his.append(train_accuracy)

            test_logits = np.dot(np.maximum(0, np.dot(test_images, W.T)), v)
            test_predictions = np.argmax(test_logits, axis=1)
            test_accuracy = np.mean(test_predictions == test_labels)
            test_accuracy_his.append(test_accuracy)

            if (i+1) % 100 == 0:
                print(f"Iteration {i+1}/{num_batches} - Training accuracy: {train_accuracy:.4f} - Test accuracy: {test_accuracy:.4f}")

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds")

    return train_accuracy_his, test_accuracy_his

def plot_accuracy(train_accuracy_his, test_accuracy_his):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(train_accuracy_his, label='Train')
    ax.plot(test_accuracy_his, label='Test')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy History')
    ax.legend()
    plt.show()

download_data()
train_images, train_labels, test_images, test_labels = load_data()
# normalize the data
train_images = normalize(train_images)
test_images = normalize(test_images)

# Convert labels to binary
train_labels = convert_labels(train_labels)
test_labels = convert_labels(test_labels)

k_sets = [5, 40, 200]
for k in k_sets:
  start_time = time.time()
  train_acc, test_acc = logistic_classifier(train_images, train_labels, test_images, test_labels, k)
  end_time = time.time()
  elapsed_time = end_time - start_time
  plot_accuracy(train_acc, test_acc)
  print("Training time (K Size = {}): {:.2f} seconds".format(k, elapsed_time))