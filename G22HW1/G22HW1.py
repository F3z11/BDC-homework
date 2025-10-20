import math
import numpy as np
from collections import Counter, defaultdict
from pyspark import SparkConf, SparkContext
from pyspark.mllib.clustering import KMeans
import sys
import os

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

def standard_distances(tup, C):
    """
    Computes the squared Euclidean distance between a point and the closest center in C.
    
    Returns:
        tuple: (point, minimum squared distance to a center)
    """
    return (tup[0], min([(math.dist(tup[0], center))**2 for center in C]))

def fair_distances(tup, C):
    """
    Computes the squared Euclidean distance between a point and the closest center in C,
    and returns the group label instead of the point.
    
    Returns:
        tuple: (group label, minimum squared distance to a center)
    """
    return (tup[1], min([(math.dist(tup[0], center))**2 for center in C]))

def min_cluster(tup, C):
    """
    Finds the cluster center closest to the given point and assigns it to that cluster.
    
    Returns:
        tuple: (index of the closest cluster, group label of the point)
    """
    distances = [math.dist(tup[0], center)**2 for center in C]
    cluster_index = int(np.argmin(distances))
    return (cluster_index, tup[1])

def MRComputeStandardObjective(U, C):
    """
    Computes the standard k-means objective function.
    
    Returns:
        float: The value of the objective function.
    """
    N = U.count()
    loss = (U.map(lambda x: standard_distances(x, C))   # MAP PHASE (R1)
              .map(lambda x: (0, x[1]))                 # REDUCE PHASE (R1)
              .groupByKey()                             # GROUPING + SHUFFLE
              .mapValues(lambda vals: sum(vals)))       # REDUCE PHASE (R2)
    return round(loss.collect()[0][1] / N, 6)

def MRComputeFairObjective(U, C):
    """
    Computes a fairness-aware objective function, where the loss is calculated separately
    for each group (A and B) and the final value is the maximum loss among them.
    
    Returns:
        float: The value of the fairness-aware objective function.
    """
    count_class = U.map(lambda point: (point[-1], 1)).reduceByKey(lambda a, b: a + b)
    count_list = count_class.collect()
    Na = count_list[0][1]
    Nb = count_list[1][1]
    
    loss = (U.map(lambda x: fair_distances(x, C))       # MAP PHASE (R1)
              .groupByKey()                             # GROUPING + SHUFFLE
              .mapValues(lambda vals: sum(vals)))       # REDUCE PHASE (R1)
    
    loss_groups = loss.collect()
    loss_A = loss_groups[0][1] / Na
    loss_B = loss_groups[1][1] / Nb
    
    return round(max(loss_A, loss_B), 6)

def MRPrintStatistics(U, C):
    """
    Computes and prints the number of points from each group assigned to each cluster.
    
    The output format is:
        i = <cluster index>, center = <cluster coordinates>, NA<i> = <count A>, NB<i> = <count B>
    """
    stats = (U.map(lambda x: min_cluster(x, C))                  # MAP PHASE (R1)
              .groupByKey()                                      # GROUPING + SHUFFLE
              .mapValues(lambda values: dict(Counter(values))))  # REDUCE PHASE (R1)
    
    for el in sorted(stats.collect()):
        counts = defaultdict(int, el[1])
        print(
            f"i = {el[0]}, center = {tuple(round(float(x),6) for x in C[el[0]])}, NA{el[0]} = {counts['A']}, NB{el[0]} = {counts['B']}"
        )

def main():

    assert len(sys.argv) == 5, "Usage: python G22HW1.py <file_name> <L> <K> <M>"

    conf = SparkConf().setAppName("kmeans")
    sc = SparkContext(conf=conf)

    L = sys.argv[2] #num of partitions
    assert L.isdigit(), "L must be an integer"
    L = int(L)

    data_path = sys.argv[1]  # This is the number 4 I add when executing the file
    assert os.path.isfile(data_path), "File or folder not found"

    K = sys.argv[3]  # num of clusters
    assert K.isdigit(), "K must be an integer"
    K = int(K)

    M = sys.argv[4] #num of iterations
    assert M.isdigit(), "M must be an integer"
    M = int(M)

    rdd = sc.textFile(data_path).cache()
    rdd = rdd.coalesce(numPartitions=L).map(parse_line) #use coalesce to avoid data shuffling when creating partitions

    N = rdd.count() #save total number of points as a global variable

    #count number of elements for each class
    count_class = rdd.map(lambda point: (point[-1], 1)).reduceByKey(lambda a, b: a+b)

    count_list = count_class.collectAsMap()
    Na = count_list['A']
    Nb = count_list['B']

    #train the kmeans model to find clusters
    rdd_points = rdd.map(lambda x:x[:-1]) #remove the class label to run the kmeans algorithm
    clusters = KMeans.train(rdd_points, k=K, maxIterations=M)

    C = clusters.clusterCenters

    print(f"Input file = {data_path}, L = {L}, K = {K}, M = {M}")
    print(f"N = {N}, NA = {Na}, NB = {Nb}")
    print(f"Delta(U,C) = {MRComputeStandardObjective(rdd, C)}") #print objective function
    print(f"Phi(A,B,C) = {MRComputeFairObjective(rdd, C)}")
    MRPrintStatistics(rdd, C)

if __name__ == "__main__":
    main()