import operator
import numpy as np
from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids


class Hinov_mod:
    """
    Read more about HINoV and modification of HINoV method:
    Frank Carmone, Ali Kara, and Sarah Maxwell
    Hinov: A new model to improve market segment definition by identifying noisy variables

    Marek Walesiak and Andrzej Dudek.
    Identification of Noisy Variables for Nonmetric and Symbolic Data in Cluster Analysis
    """

    def __init__(self, ds):
        """
        :param ds: pandas DataFrame structure to perform feature selection on.
        """
        self.dataset = ds
        self.number_of_variables = len(ds.columns)
        self.number_of_instances = len(ds)

    def topri(self, var_clusters):
        """
        Method that computes parim matrix.
        :param var_clusters: Numpy array of ndarrays, value returned by perform_kmeans method.
        :return: TOPRI matrix of of number_of_variables x 1 shape, numpy.ndarray type.
        """
        self.PARIM_matrix = np.empty(shape=(self.number_of_variables, self.number_of_variables))
        for i in range(self.number_of_variables):
            for j in range(self.number_of_variables):
                if (i == j):
                    self.PARIM_matrix[i, j] = 0
                else:
                    self.PARIM_matrix[i, j] = adjusted_rand_score(var_clusters[i], var_clusters[j])
        self.TOPRI_matrix = np.sum(self.PARIM_matrix, axis=1)
        return self.TOPRI_matrix


    def scree_plot(self, topri_m):
        """
        Method that sorts variables by their TOPRI values and then displays a scree plot to select relevant
        features from.
        :param topri_m: topri matrix as a numpy array of ndarrays, value returned by topri method.
        :return: list of variables sorted by their TOPRI values and list of corresponding TOPRI values.
        """
        self.TOPRI_dict = {}
        for i in range(self.number_of_variables):
            self.TOPRI_dict[str(i + 1)] = topri_m[i]
        self.TOPRI_sorted = sorted(self.TOPRI_dict.items(), key=operator.itemgetter(1), reverse=True)
        self.variables_sorted = [self.TOPRI_sorted[i][0] for i in range(self.number_of_variables)]
        self.TOPRI_values_sorted = [self.TOPRI_sorted[i][1] for i in range(self.number_of_variables)]
        plt.plot(self.variables_sorted, self.TOPRI_values_sorted, '-o')
        plt.grid(True)
        plt.show()
        return self.variables_sorted, self.TOPRI_values_sorted

    def auto_select(self, topri_m):
        """
        Method that automatically selects relevant features based on K-means clustering performed on TOPRI matrix.
        An alternative for the subjective scree plot method.
        :param topri_m: topri matrix as a numpy array of ndarrays, value returned by topri method.
        :return: array of relevant features.
        """
        max_index = list(topri_m).index(max(topri_m))
        self.selection = cluster.KMeans(n_clusters=2, n_init=1, init=np.array([[min(topri_m)], [max(topri_m)]]))
        topri_m=topri_m.reshape(-1,1)
        self.selection.fit(topri_m)
        self.relevant = self.selection.predict(topri_m)
        if(self.relevant[max_index]==0):
            for i in range(self.number_of_variables):
                self.relevant[i]=self.relevant[i]-1
        return(self.relevant)



class Hinov_mod_part(Hinov_mod):
    """
    Implementation of HINoV method using KMeans algorithm. A child of Hinov class.
    """
    def __init__(self, ds):
        super().__init__(ds)

    def perform_kmeans(self, n_clust):
        """
        Method that builds and then trains KMeans models for each single variable from the dataset.
        :param n_clust: Number of clusters to form.
        :return: Array of ndarrays. Each ndarray represents index of the cluster each sample belongs to.
        Read more: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        """
        self.model = cluster.KMeans(n_clusters=n_clust)
        self.matrices = np.empty((self.number_of_variables, self.number_of_instances))
        i = 0
        for column in self.dataset.columns:
            self.variable_array = np.array(self.dataset[column]).reshape(-1,
                                                                         1)  # clustering on a single feature, so reshape(-1,1) is needed
            self.model.fit(self.variable_array)
            self.matrices[i] = np.array(self.model.predict(self.variable_array))
            i += 1
        return self.matrices

    def perform_PAM(self, n_clust):
        """
         Method that builds and then trains PAM models for each single variable from the dataset.
        :param n_clust: Number of clusters to form.
        :return: Array of ndarrays. Each ndarray represents index of the cluster each sample belongs to.
        Read more: https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html
        """
        self.model = KMedoids(n_clusters=n_clust, method='pam')
        self.matrices = np.empty((self.number_of_variables, self.number_of_instances))
        i = 0
        for column in self.dataset.columns:
            self.variable_array = np.array(self.dataset[column]).reshape(-1,
                                                                         1)  # clustering on a single feature, so reshape(-1,1) is needed
            self.model.fit(self.variable_array)
            self.matrices[i] = np.array(self.model.predict(self.variable_array))
            i += 1
        print(type(self.matrices))
        print(self.matrices)
        print(self.matrices.shape)
        return self.matrices

class Hinov_mod_hierarchical(Hinov_mod):
    """
    Implementation of HINoV method using hierarchical algorithms. A child of Hinov class.
    """
    def __init__(self, ds):
        super().__init__(ds)

    def perform_hierarchical(self, n_clust=2, alg='ward'):
        """
        Method that builds and then trains hierarchical clustering models using selected linkage for each single
        variable from the dataset.
        :param n_clust: The number of clusters to find. Default: 2.
        :param alg: Possible values: ‘ward’, ‘complete’, ‘average’, ‘single’. Default: 'ward'.
        :return: Array of ndarrays. Each ndarray represents index of the cluster each sample belongs to.
        Read more: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering
        """
        self.model = cluster.AgglomerativeClustering(n_clusters=n_clust, linkage=alg)
        self.matrices = np.empty((self.number_of_variables, self.number_of_instances))
        i = 0
        for column in self.dataset.columns:
            self.variable_array = np.array(self.dataset[column]).reshape(-1,
                                                                         1)  # clustering on a single feature, so reshape(-1,1) is needed
            self.matrices[i] = np.array(self.model.fit_predict(self.variable_array))
            i += 1
        return self.matrices

