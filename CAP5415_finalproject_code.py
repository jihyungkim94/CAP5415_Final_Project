import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA, KernelPCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import mode


def get_randomized_features_subsets(labels, num_features):
    return np.random.choice(a = labels,
                          size = [int(len(labels) / num_features), num_features],
                          replace = False)


def make_training_subset(X_bootstrap, y_bootstrap, features_subset):
    X_pca_subset, y_pca_subset = [], []
    for i in range(len(features_subset)):
        X_pca_subset = np.concatenate((X_pca_subset, X_bootstrap[np.where(y_bootstrap == features_subset[i])]))
        #y_pca_subset = np.concatenate((y_pca_subset, y_bootstrap[np.where(y_bootstrap == features_subset[i])]))
    dim_X, _ = train_test_split(list(range(len(X_pca_subset))), train_size = 100 * 100, shuffle = True)
    return X_pca_subset[dim_X]#, y_pca_subset[dim_y]


def rotation_tree(X_Train, y_Train, num_trees, num_features):
    forest = []
    rotation_matrices = []
    features = []

    for i in range(num_trees):
        if (i % 2 == 0):
            print("Training Progress: ", np.round(((i + 1) / num_trees * 100.0), 1), "%")

        # bootstrap 75% of training data
        X_bootstrap, _, y_bootstrap, _ = train_test_split(X_Train, y_Train, train_size = 0.75, shuffle = True)

        # get feature subsets
        randomized_features_subsets = get_randomized_features_subsets(np.unique(y), num_features)
        features.append(randomized_features_subsets)

        for features_subset in randomized_features_subsets:
            # get observations corresponding to features subset
            X_pca_subset = make_training_subset(X_bootstrap, y_bootstrap, features_subset)
            X_pca_subset = np.reshape(X_pca_subset, (-1, int(np.sqrt(X_pca_subset.shape[0]))))
            #y_pca_subset = np.reshape(y_pca_subset, (-1, int(np.sqrt(y_pca_subset.shape[0]))))

            #print(X_pca_subset.shape)

            # apply linear PCA or kernel PCA
            #pca = PCA(whiten = True, n_components = num_features)
            pca = KernelPCA(n_components = num_features, kernel = 'rbf', n_jobs = -1)

            pca.fit(X_pca_subset)

            # fill rotation matrix diagonal with principal components
            #rotation_matrix = np.zeros((X_pca_subset.shape[0], X_pca_subset.shape[0]), dtype = float)
            #np.fill_diagonal(a = rotation_matrix, val = pca.components_[0])

            rotation_matrix = np.zeros((X_pca_subset.shape[0], X_pca_subset.shape[0]), dtype = float)
            np.fill_diagonal(a = rotation_matrix, val = pca.eigenvalues_[:])

            rotation_matrices.append(rotation_matrix)

            X_train_subset, _, y_train_subset, _ = train_test_split(X_bootstrap, y_bootstrap,
                                                                    train_size = 100 * 100, shuffle = True)
            X_train_subset = np.reshape(X_train_subset, (-1, int(np.sqrt(X_train_subset.shape[0]))))

            # apply rotation transformation
            X_train_subset_transformed = np.dot(X_train_subset, rotation_matrix)

            X_train_subset_transformed = np.reshape(X_train_subset_transformed,
                                                    (X_train_subset_transformed.shape[0] * X_train_subset_transformed.shape[1],
                                                     -1))

            tree = DecisionTreeClassifier(criterion = "entropy", splitter = "best").fit(X_train_subset_transformed,
                                                                                        y_train_subset)
            forest.append(tree)

    return forest, rotation_matrices, features

def forest_pred(forest, rotation_matrices, X_Test, y_Test):
    predictions = []
    for i in range(len(forest)):
        X_test_transformed = np.dot(X_Test, rotation_matrices[i])
        X_test_transformed = np.reshape(X_test_transformed, (len(y_Test), -1))
        predictions.append(forest[i].predict(X_test_transformed))
    predictions = np.asarray(predictions)
    output = mode(predictions)[0].flatten()

    return accuracy_score(y_true = y_Test, y_pred = output)


if __name__ == '__main__':
    num_trees = 100
    num_features = 4

    X = np.load('ip_data.npy')
    y = np.load('ip_label.npy')

    newX = np.reshape(X, (-1, X.shape[2]))
    # scale to unit variance
    newX = StandardScaler().fit_transform(newX)

    # label cloning as spectral bands have been fanned apart
    y = np.repeat(a = y[:, :, np.newaxis], repeats = newX.shape[1], axis = 2)

    X = np.reshape(newX, (newX.shape[0] * newX.shape[1], -1))
    y = np.reshape(y, (y.shape[0] * y.shape[1] * y.shape[2], -1))

    # resample data to address label imbalances
    X, y = SMOTE().fit_resample(X, y)
    X = X.flatten()

    # reserve 30% data for testing
    X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size = 0.3, shuffle = True,
                                                    random_state = 25)

    forest, rotation_matrices, features = rotation_tree(X_Train, y_Train, num_trees, num_features)

    cv_accuracy = np.zeros(5)
    for i in range(5):
        X_sample, _, y_sample, _ = train_test_split(X_Test, y_Test, train_size = 10000, shuffle = True)
        X_sample = np.reshape(X_sample, (-1, int(np.sqrt(X_sample.shape[0]))))
        y_sample = np.reshape(y_sample, (y_sample.shape[0], 1))
        cv_accuracy[i] = forest_pred(forest, rotation_matrices, X_sample, y_sample)

    print(np.round(np.mean(cv_accuracy) * 100.0, 2))
    print(np.round(np.std(cv_accuracy) * 100.0, 4))


