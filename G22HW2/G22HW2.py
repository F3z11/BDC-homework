import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.mllib.clustering import KMeans
import sys
import time

def parse_line(line):
    """
    Parses a line from the input file.
    Each line is expected to be a comma-separated list of values.
    The last value represents the group label, while the preceding values represent a point in space.
    
    Returns:
        tuple: (point as tuple of floats, group label as string)
    """
    parts = line.split(",")
    point = tuple(map(float, parts[:-1]))
    group = parts[-1]
    return (point, group)

def min_cluster(tup, C):
    """
    Assigns a point to the nearest cluster based on Euclidean distance (squared).
    
    Returns:
        tuple: (index of the nearest cluster centroid, original input tuple)
    """
    diffs = np.array(C) - np.array(tup[0])
    distances = np.sum(diffs**2, axis=1)
    cluster_index = int(np.argmin(distances))
    return (cluster_index, tup)

def computeVectorX(fixed_a, fixed_b, alpha, beta, ell, k):
    gamma = 0.5
    x_dist = [0.0] * k
    power = 0.5
    t_max = 10

    for _ in range(t_max):
        f_a = fixed_a
        f_b = fixed_b
        power /= 2

        for i in range(k):
            temp = (1 - gamma) * beta[i] * ell[i] / (gamma * alpha[i] + (1 - gamma) * beta[i])
            x_dist[i] = temp
            f_a += alpha[i] * temp * temp
            temp = ell[i] - temp
            f_b += beta[i] * temp * temp

        if f_a == f_b:
            break

        gamma = gamma + power if f_a > f_b else gamma - power

    return x_dist

def seq_op(acc, point_tuple):
    """
    Accumulates statistics for a single point within a partition.

    Returns:
        tuple: Updated accumulated values after processing the given point.
    """
    count_a, count_b, sum_a, sum_b, sum_sq_a, sum_sq_b = acc
    point, label = point_tuple
    point_array = np.array(point)
    
    if label == 'A':
        count_a += 1
        sum_a = sum_a + point_array
        sum_sq_a += np.sum(point_array**2)
    else:  # label == 'B'
        count_b += 1
        sum_b = sum_b + point_array
        sum_sq_b += np.sum(point_array**2)
    
    return (count_a, count_b, sum_a, sum_b, sum_sq_a, sum_sq_b)

def comb_op(acc1, acc2):
    """
    Combines two accumulators from different partitions into a single accumulator.

    Returns:
        tuple: Combined accumulator with corresponding values summed.
    """
    count_a1, count_b1, sum_a1, sum_b1, sum_sq_a1, sum_sq_b1 = acc1
    count_a2, count_b2, sum_a2, sum_b2, sum_sq_a2, sum_sq_b2 = acc2
    
    return (
        count_a1 + count_a2,
        count_b1 + count_b2,
        sum_a1 + sum_a2,
        sum_b1 + sum_b2,
        sum_sq_a1 + sum_sq_a2,
        sum_sq_b1 + sum_sq_b2
    )

def compute_params(kv, zero_vec):
    """
    Computes cluster-specific parameters based on accumulated statistics for a cluster.

    Returns:
        tuple: (alpha_i, beta_i, mu_a_i, mu_b_i, l_i, delta_a_i, delta_b_i)
            - alpha_i (float): Proportion of 'A' points in the cluster relative to total 'A' points.
            - beta_i (float): Proportion of 'B' points in the cluster relative to total 'B' points.
            - mu_a_i (numpy array): Mean vector of points labeled 'A' in the cluster.
            - mu_b_i (numpy array): Mean vector of points labeled 'B' in the cluster.
            - l_i (float): Euclidean distance between mu_a_i and mu_b_i.
            - delta_a_i (float): Sum of squared deviations for 'A' points in the cluster.
            - delta_b_i (float): Sum of squared deviations for 'B' points in the cluster.
    """
    cluster_idx, (count_a, count_b, sum_a, sum_b, sum_sq_a, sum_sq_b) = kv
    
    alpha_i = count_a / Na if Na > 0 else 0.0
    beta_i = count_b / Nb if Nb > 0 else 0.0
    
    if count_a > 0:
        mu_a_i = sum_a / count_a
        delta_a_i = sum_sq_a - np.sum(mu_a_i**2) * count_a
    else:
        mu_a_i = zero_vec
        delta_a_i = 0.0
    
    if count_b > 0:
        mu_b_i = sum_b / count_b
        delta_b_i = sum_sq_b - np.sum(mu_b_i**2) * count_b
    else:
        mu_b_i = zero_vec
        delta_b_i = 0.0
    
    if count_a == 0:
        mu_a_i = mu_b_i
    if count_b == 0:
        mu_b_i = mu_a_i
    
    l_i = np.linalg.norm(mu_a_i - mu_b_i)
    
    return (alpha_i, beta_i, mu_a_i, mu_b_i, l_i, delta_a_i, delta_b_i)

def CentroidSelection(components):
    """
    Computes new cluster centroids based on the computed parameters for each cluster.
    
    Returns:
        list of numpy arrays: New centroids for each cluster.
    """
    def compute_c(i):
        """
        Computes the new centroid position for cluster i.
        
        Returns:
            numpy array: New centroid position for cluster i.
        """
        return ((l[i] - x[i]) * M_a[i] + x[i] * M_b[i]) / l[i]

    alpha, beta, M_a, M_b, l, delta_a_list, delta_b_list = zip(*components)

    fixed_a = sum(delta_a_list) / Na if Na > 0 else 0.0
    fixed_b = sum(delta_b_list) / Nb if Nb > 0 else 0.0
    k = len(alpha)

    x = computeVectorX(fixed_a, fixed_b, alpha, beta, l, k)

    C = []
    for i in range(k):
        if l[i] == 0:
            C.append(M_a[i])
        else:
            C.append(compute_c(i))

    return C

def MRFairLloyd(U, K, M):
    """
    Performs a fair clustering using a fair version of Lloyd's algorithm
    
    Returns:
        list of numpy arrays: Final cluster centroids after M iterations.
    """
    rdd_points = U.map(lambda x: x[:-1])
    clusters = KMeans.train(rdd_points, k=K, maxIterations=0)
    C = clusters.clusterCenters
    
    for i in range(M):
        
        dim = len(U.first()[0])
        zero_vec = np.zeros(dim)
        
        # Define initial accumulator for aggregateByKey
        # Format: (count_A, count_B, sum_A_points, sum_B_points, sum_A_squared, sum_B_squared)
        init_acc = (0, 0, zero_vec, zero_vec, 0.0, 0.0)
        
        # Compute stats by cluster within partition (seq_op) and then aggregate across partitions (comb_op)
        
        params = (U.map(lambda x: min_cluster(x, C))  # MAP R1
                    .aggregateByKey(init_acc, seq_op, comb_op) # REDUCE R1
                    .map(lambda kv: compute_params(kv, zero_vec))) # REDUCE R2
        
        params = params.collect()
        
        C = CentroidSelection(params)
    
    return C

def fair_distances(tup, C):
    """
    Computes the squared Euclidean distance between a point and its closest cluster center.
    
    Returns:
        tuple: (label, min_dist_sq)
            - label: same as input label,
            - min_dist_sq: minimum squared Euclidean distance from the point to any center in C.
    """
    point = np.array(tup[0])
    centers = np.array(C)
    diffs = centers - point
    dists_sq = np.einsum('ij,ij->i', diffs, diffs)
    min_dist_sq = np.min(dists_sq)
    return (tup[1], min_dist_sq)


def MRComputeFairObjective(U, C):
    """
    Computes a fairness-aware objective function, where the loss is calculated separately
    for each group (A and B) and the final value is the maximum loss among them.
    
    Returns:
        float: The value of the fairness-aware objective function.
    """

    loss = (U.map(lambda x: fair_distances(x, C))
            .reduceByKey(lambda x, y: x + y))
    
    loss_groups = dict(loss.collect())

    loss_A = loss_groups.get('A', 0) / Na if Na > 0 else 0.0
    loss_B = loss_groups.get('B', 0) / Nb if Nb > 0 else 0.0

    return round(max(loss_A, loss_B), 4)

def main():

    assert len(sys.argv) == 5, "Usage: python G22HW2.py <file_name> <L> <K> <M>"

    conf = SparkConf().setAppName("kmeans")
    sc = SparkContext(conf=conf)

    L = sys.argv[2] #num of partitions
    assert L.isdigit(), "L must be an integer"
    L = int(L)

    data_path = sys.argv[1]

    K = sys.argv[3]  # num of clusters
    assert K.isdigit(), "K must be an integer"
    K = int(K)

    M = sys.argv[4] #num of iterations
    assert M.isdigit(), "M must be an integer"
    M = int(M)

    inputPoints = sc.textFile(data_path).cache()
    inputPoints = inputPoints.repartition(numPartitions=L).map(parse_line)

    global N, Na, Nb
    N = inputPoints.count() #save total number of points as a global variable

    #count number of elements for each class
    count_class = inputPoints.map(lambda point: (point[-1], 1)).reduceByKey(lambda a, b: a+b)

    count_list = count_class.collectAsMap()
    Na = count_list.get('A', 0)
    Nb = count_list.get('B', 0)

    #train the kmeans model to find clusters
    rdd_points = inputPoints.map(lambda x:x[:-1]) #remove the class label to run the standard kmeans algorithm

    time_stand_start = time.time()
    clusters_stand = KMeans.train(rdd_points, k=K, maxIterations=M)
    C_stand = clusters_stand.clusterCenters
    time_stand_end = time.time()

    time_stand = (time_stand_end - time_stand_start) * 1000

    time_fair_start = time.time()
    C_fair = MRFairLloyd(inputPoints, K, M)
    time_fair_end = time.time()

    time_fair = (time_fair_end - time_fair_start) * 1000

    time_stand_loss_start = time.time()
    loss_fair_cstand = MRComputeFairObjective(inputPoints, C_stand)
    time_stand_loss_end = time.time()

    time_stand_loss = (time_stand_loss_end - time_stand_loss_start) * 1000

    time_fair_loss_start = time.time()
    loss_fair_cfair = MRComputeFairObjective(inputPoints, C_fair)
    time_fair_loss_end = time.time()

    time_fair_loss = (time_fair_loss_end - time_fair_loss_start) * 1000


    print(f"Input file = {data_path}, L = {L}, K = {K}, M = {M}")
    print(f"N = {N}, NA = {Na}, NB = {Nb}")

    print(f'Fair Objective with Standard Centers = {loss_fair_cstand}')
    print(f'Fair Objective with Fair Centers = {loss_fair_cfair}')

    print(f'Time to compute standard centers = {int(time_stand)} ms')
    print(f'Time to compute fair centers = {int(time_fair)} ms')

    print(f'Time to compute objective with standard centers = {int(time_stand_loss)} ms')
    print(f'Time to compute objective with fair centers = {int(time_fair_loss)} ms')

if __name__ == "__main__":
    main()