import struct as st
import numpy as np
import matplotlib.pyplot as plt 


PI = 3.14


# Unpacks the binary files and sets up the arrays based on
# file format that can be found here: http://yann.lecun.com/exdb/mnist/ .
def Unpack_bins():

    # Simple linear struct unpack to get the bytes of the two files into NumPy arrays.

    training_images_file = open('../Resources/train-images-idx3-ubyte', 'rb')
    training_images_file.seek(0)
    magic_num = st.unpack('>4B', training_images_file.read(4))
    num_of_training_images = st.unpack('>I', training_images_file.read(4))[0]
    num_of_training_rows = st.unpack('>I', training_images_file.read(4))[0]
    num_of_training_cols = st.unpack('>I', training_images_file.read(4))[0]
    training_images_array = []

    training_labels_file = open('../Resources/train-labels-idx1-ubyte', 'rb')
    training_labels_file.seek(0)
    training_labels_magic_num = st.unpack('>4B', training_labels_file.read(4))
    num_of_labels = st.unpack('>I', training_labels_file.read(4))[0]
    training_labels_array = []

    for i in range(0, 60000):
        label = st.unpack('>' + 'B', training_labels_file.read(1))[0]
        image = []
        bits = num_of_training_rows * num_of_training_cols
        for j in range(0, bits):
            pixel = st.unpack('>' + 'B', training_images_file.read(1))[0]
            image.append(pixel)
        if label < 4:
            training_labels_array.append(label)
            training_images_array.append(image)

    testing_images_file = open('../Resources/t10k-images-idx3-ubyte', 'rb')
    testing_images_file.seek(0)
    testing_images_magic_num = st.unpack('>4B', testing_images_file.read(4))
    num_of_testing_images = st.unpack('>I', testing_images_file.read(4))[0]
    num_of_testing_rows = st.unpack('>I', testing_images_file.read(4))[0]
    num_of_testing_cols = st.unpack('>I', testing_images_file.read(4))[0]
    testing_images_array = []

    testing_labels_file = open('../Resources/t10k-labels-idx1-ubyte', 'rb')
    testing_labels_file.seek(0)
    testing_labels_magic_num = st.unpack('>4B', testing_labels_file.read(4))
    num_of_testing_labels = st.unpack('>I', testing_labels_file.read(4))[0]
    testing_labels_array = []

    for i in range(0, 10000):
        label = st.unpack('>' + 'B', testing_labels_file.read(1))[0]
        image = []
        bits = num_of_training_rows * num_of_training_cols
        for j in range(0, bits):
            pixel = st.unpack('>' + 'B', testing_images_file.read(1))[0]
            image.append(pixel)
        if label < 4:
            testing_labels_array.append(label)
            testing_images_array.append(image)

    return np.array(training_images_array), np.array(training_labels_array), np.array(testing_images_array), np.array(
        testing_labels_array)


# Initialize X_transformed array based on average brightnesses of array's X rows and columns.
def init_trans(X):
    # Reshape every samples' features into a 28x28 image.
    X = X.reshape(X.shape[0], 28, 28)
    X_transformed = np.zeros((X.shape[0], 2))
    # For every sample.
    for k in range(0, X.shape[0]):
        row_sum = np.zeros(X.shape[1])
        # Find Average Row Brightness.
        for i in range(0, X.shape[1]):
            for j in range(0, X.shape[2]):
                row_sum[i] += X[k][i][j]
            row_sum[i] /= X.shape[1]
        for i in range(0, X.shape[1]):
            X_transformed[k][0] += row_sum[i] / X.shape[1]
        # Find Average Column Brightness.
        col_sum = np.zeros(X.shape[2])
        for j in range(0, X.shape[2]):
            for i in range(0, X.shape[1]):
                col_sum[j] += X[k][i][j]
            col_sum[j] /= X.shape[2]
        for j in range(0, X.shape[2]):
            X_transformed[k][1] += col_sum[j] / X.shape[2]
    return X_transformed


# Plots points of Χ based on the classes represented by the array labels.
def plot(X, labels):
    color = ['red', 'green', 'cyan', 'yellow']
    colors = np.empty(len(X), dtype=object)
    for i in range(0, len(X)):
        colors[i] = color[labels[i]]
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    plt.show()


# My implementation of Maximin algorithm for K centers initialization.
def Maximin(X, K):

    [num_of_samples, num_of_features] = X.shape
    centers = np.zeros((K, num_of_features))
    points = X
    distances = np.zeros((num_of_samples, 1))

    # Choose 1st center as the one with the smallest distance from 0.
    for i in range(0, num_of_samples):
        distances[i] = np.linalg.norm(points[i] - 0)
    min_index = np.argmin(distances)
    centers[0] = points[min_index]
    points = np.delete(points, min_index, axis=0)

    # Find distances from the 1st center.
    for i in range(0, points.shape[0]):
        distances[i] = np.linalg.norm(points[i] - centers[0])
    max_index = np.argmax(distances)

    # Choose the most distant point from the 1st center, as the 2nd center.
    centers[1] = points[max_index]
    points = np.delete(points, max_index, axis=0)

    # Choose the rest centers.
    for i in range(2, K):
        distances = np.zeros((points.shape[0], i))
        for j in range(0, i):
            for k in range(0, len(distances)):
                distances[k][j] = np.linalg.norm(points[k] - centers[j])
        min_dist = np.zeros(len(distances))
        for j in range(0, len(distances)):
            min_dist[j] = min(distances[j])
        max_dist_index = np.argmax(min_dist)
        centers[i] = points[max_dist_index]
        points = np.delete(points, max_dist_index, axis=0)
    return centers


# My implementation of K-means algorithm for K centers.
def K_means(X, K):

    [num_of_samples, num_of_features] = X.shape
    centers = Maximin(X, K)
    clusters = np.full(num_of_samples, 0)
    distances = np.zeros((K, num_of_samples))
    while True:
        prev_centers = centers.copy()

        # Compute distance between every cluster center and sample.
        for k in range(0, K):
            for j in range(0, num_of_samples):
                distances[k][j] = np.linalg.norm(X[j] - centers[k])

        # Pick a cluster for each sample.
        for j in range(0, num_of_samples):
            clusters[j] = np.argmin(distances[:, j])

        # Update center of each cluster.
        for j in range(0, K):
            cluster = X[clusters == j]
            centers[j] = cluster.mean(axis=0)

        # Break the loop if the centers are not changed.
        if (prev_centers == centers).all():
            break

    return clusters


# Calculates the purity of a cluster.
def purity(X, labels, K, cls):
    purity_sum = 0
    # For each cluster.
    for i in range(0, K):
        class_count = np.full(K, 0)
        # Find the label with the most participants in a cluster and divide that by the cluster size.
        for j in range(0, len(X)):
            if cls[j] == i:
                class_count[labels[j]] += 1
        for j in range(0, len(X)):
            if cls[j] == i and labels[j] == class_count.argmax():
                purity_sum += 1
    return purity_sum / len(X)


# My implementation of PCA algorithm for reduction to V dimensions.
def PCA(X, dimensions):
    # Mean Normalize M.
    mean = np.mean(X, 0)
    normalized_X = X - mean

    # Calculate Covariance Matrix.
    covariance_matrix = np.cov(normalized_X.T)

    #  Find eigenValues and eigenVectors.
    eigenValues, eigenVectors = np.linalg.eig(covariance_matrix)

    # Sort eigenValues and eigenVectors.
    indexes = eigenValues.argsort()[::-1]
    eigenVectors = eigenVectors[:, indexes]

    # Find Reduced Dimensions Matrix.
    U_reduced = eigenVectors[:, :dimensions]

    # Calculate transformed X.
    X_transformed = np.dot(normalized_X, U_reduced).real

    # multiply by -1 for visualization purposes.
    return -1 * X_transformed


# Calculate the probability that a sample belongs to each class.
def prior_probability(labels, K):
    prior = np.zeros(K)
    for i in range(0, labels.shape[0]):
        prior[labels[i]] += 1
    return prior / labels.shape


# Calculate product of conditional probabilities using gaussian distribution for each sample of X
# in order to calculate the possibility that a sample belongs to a specific class.
def posterior_probability(X, m, var, K):
    [num_of_samples, num_of_features] = X.shape
    post_prob = np.ones((K, num_of_samples))
    for i in range(0, num_of_samples):
        for k in range(0, K):
            for j in range(0, num_of_features):
                post_prob[k][i] *= (1 / np.sqrt(2 * PI * var[k][j])) * np.exp(-0.5 * pow((X[i][j] - m[k][j]), 2) / var[k][j])
    return post_prob


# My implementation of Gaussian Naive Bayes Estimator
def GNB(training_set, training_labels, testing_set, K):
    [num_of_training_samples, num_of_training_sample_features] = training_set.shape
    [num_of_testing_samples, num_of_testing_sample_features] = testing_set.shape

    # Calculate mean and variance array for each feature based on the training set.
    mean = np.zeros((K, num_of_training_sample_features))
    variance = np.zeros((K, num_of_training_sample_features))
    for i in range(0, K):
        cluster = training_set[L_tr == i]
        mean[i] = cluster.mean(axis=0)
        variance[i] = cluster.var(axis=0)

    # Calculate probability that a train sample belongs to a class.
    classes = np.full(num_of_testing_samples, 0)
    prior_prob = prior_probability(training_labels, K)
    posterior_prob = posterior_probability(testing_set, mean, variance, K)

    # Choose the most probable class for each train sample and assign it to that sample.
    for i in range(0, num_of_testing_samples):
        class_prob = np.ones(K)
        total_prob = 0
        for k in range(0, K):
            total_prob += posterior_prob[k][i] * prior_prob[k]

        for k in range(0, K):
            class_prob[k] = (posterior_prob[k][i] * prior_prob[k]) / total_prob

        classes[i] = class_prob.argmax()

    return classes


if __name__ == '__main__':
    # Part 1: Initialize the arrays containing the train/test sets and train/test labels.
    M, L_tr, N, L_te = Unpack_bins()

    # Part 2: Initialize M̂ based on average brightness of M set and plot the points of M̂.
    M̂ = init_trans(M)
    plot(M̂, L_tr)
    # Part 3: Run the K-Mean algorithm for K = 4 and plot the cluster results.
    K = 4
    classes = K_means(M̂, K)
    plot(M̂, classes)

    # Calculate the purity of the clusters given by K-mean.
    purity_percentage = purity(M̂, L_tr, K, classes) * 100
    print("The K-means clustering has " + str(purity_percentage) + "% purity for average brightness.")

    # Part 4: Use PCA to reduce 784 features of each sample to V (where V = 2, 25, 50, 100) features and then run the K-Mean algorithm to put the samples into clusters.
    V = 2
    M̃ = PCA(M, V)
    classes = K_means(M̃, K)
    plot(M̃, classes)

    # Calculate the purity of the clusters given by K-mean after PCA.
    purity_percentage = purity(M̃, L_tr, K, classes) * 100
    print("The K-means clustering has " + str(purity_percentage) + "% purity for V = " + str(V) + ".")

    # Run for V = 25, 50, 100 and find the V that gives the most purity in the clusters returned.
    M̃_max = M̃
    max_percentage = purity_percentage
    V_max = 2
    for V in range(25, 101, 25):
        if V != 75:
            M̃ = PCA(M, V)
            classes = K_means(M̃, K)

            purity_percentage = purity(M̃, L_tr, K, classes) * 100
            print("The K-means clustering has " + str(purity_percentage) + "% purity for V = " + str(V) + ".")

            if purity_percentage > max_percentage:
                M̃_max = M̃
                max_percentage = purity_percentage
                V_max = V

    # Task 5: Train the Gaussian Naive Bayes Classifier with M̃_max samples as input and test the prediction purity for Ñ_max samples.
    Ñ_max = PCA(N, V_max)
    classes = GNB(M̃_max, L_tr, Ñ_max, K)

    purity_percentage = purity(Ñ_max, L_te, K, classes) * 100
    print("The Gaussian Naive Bayes clustering has " + str(purity_percentage) + "% purity for V = " + str(V) + ".")
