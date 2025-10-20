from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark import StorageLevel
import threading
import sys
import numpy as np
import random

def generate_hash_function(g=False):
    p = 8191
    a = random.randint(1, p-1) # for h function
    b = random.randint(0, p-1) # for h function

    a_1 = random.randint(1, p-1) # for g function
    b_1 = random.randint(0, p-1) # for g function

    if g == False:
        def hash_function(x, C):
            return ((a * x + b) % p) % C
        
        return hash_function
    else:
        def hash_function(x, C):
            return ((a * x + b) % p) % C
        
        def g(x, C=2):
            return ((((a_1 * x + b_1) % p) % C) * 2) - 1
        
        return hash_function, g

def CountMinSketch(x):
    for j, hash_func in enumerate(hash_functions_cm):
        col_index = hash_func(x, W)
        cm_matrix[j, col_index] += 1

def CountSketch(x):
    for j, (hash_func, g_func) in enumerate(hash_functions_cs):
        col_index = hash_func(x, W)
        g_value = g_func(x)
        cs_matrix[j, col_index] += g_value

def topk_hitters(exact_values, pred_values, K):
    # Get top-K values sorted descending
    top_k_values = sorted(exact_values.values(), reverse=True)
    
    threshold = top_k_values[K-1]

    # Select keys meeting the threshold
    selected_keys = [key for key, val in exact_values.items() if val >= threshold]

    topk_exact = {key: exact_values[key] for key in selected_keys}

    topk_pred = {key: pred_values[key] for key in selected_keys}

    return topk_exact, topk_pred

def avg_error(exact_values, pred_values, K):
    # Get top-K values sorted descending
    top_k_values = sorted(exact_values.values(), reverse=True)
    
    threshold = top_k_values[K-1]

    # Select keys meeting the threshold
    selected_keys = [key for key, val in exact_values.items() if val >= threshold]

    # Compute the summed relative error
    error = [(abs(exact_values[key] - pred_values[key])) / exact_values[key] for key in selected_keys]

    return sum(error) / len(error)

# Operations to perform after receiving an RDD 'batch' at time 'time'
def process_batch(time, batch):
    # We are working on the batch at time `time`.
    global streamLength, exact_count
    batch_size = batch.count()
    # If we already have enough points (> THRESHOLD), skip this batch.
    if streamLength[0]>=T:
        return
    streamLength[0] += batch_size

    # Extract the exact frequencies of all distinct items
    batch_counts = batch.map(lambda s: (int(s), 1)).reduceByKey(lambda i1, i2: i1+i2).collectAsMap()
    
    # Accumulate into global exact_count
    for key, count in batch_counts.items():
        if key in exact_count:
            exact_count[key] += count
        else:
            exact_count[key] = count

    # Count-min sketch
    for el in batch.collect():
        CountMinSketch(int(el))
        CountSketch(int(el))
            
    # If we wanted, here we could run some additional code on the global histogram
    """if batch_size > 0:
        print("Batch size at time [{0}] is: {1}".format(time, batch_size))"""

    if streamLength[0] >= T:
        stopping_condition.set() # I wake up the main thread (the streaming data) to finish the streaming processing with Spark

if __name__ == '__main__':
    assert len(sys.argv) == 6, "USAGE: portExp, T, D, W, K"

    conf = SparkConf().setMaster("local[*]").setAppName("G22HW3")

    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 0.01)  # Batch duration of 0.01 seconds
    ssc.sparkContext.setLogLevel("ERROR")

    stopping_condition = threading.Event() # Controls the execution of more threads
    
    
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # INPUT READING
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    portExp = int(sys.argv[1])

    global D, W, K, cm_matrix, cs_matrix

    T = int(sys.argv[2])

    D = int(sys.argv[3]) # Number of rows, corresponds to the number of hash functions

    W = int(sys.argv[4]) # Number of columns, values that can assume the hash functions

    K = int(sys.argv[5])
        
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # DEFINING THE REQUIRED DATA STRUCTURES TO MAINTAIN THE STATE OF THE STREAM
    # &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    
    # Exact count
    streamLength = [0]
    exact_count = {}

    # Conservative Count-min sketch
    hash_functions_cm = [generate_hash_function(g=False) for _ in range(D)]
    cm_matrix = np.zeros((D, W))

    # Count-sketch
    hash_functions_cs = [generate_hash_function(g=True) for _ in range(D)]
    cs_matrix = np.zeros((D, W))

    # CODE TO PROCESS AN UNBOUNDED STREAM OF DATA IN BATCHES
    stream = ssc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevel.MEMORY_AND_DISK)
    # For each batch, to the following.
    stream.foreachRDD(lambda time, batch: process_batch(time, batch))
    
    # MANAGING STREAMING SPARK CONTEXT
    ssc.start() # Start streaming processing
    stopping_condition.wait() # Data keeps arriving (main thread) but is waiting, but the Spark thread (second one) keeps processing data.
    
    ssc.stop(False, False) # Now the main thread is stopped definitely

    # COMPUTE AND PRINT FINAL STATISTICS

    # Conservative Count-min sketch final computation
    dict_count_min = exact_count.copy()

    for key in dict_count_min.keys():
        dict_count_min[key] = min([cm_matrix[row, hash_func(key, W)] for row, hash_func in enumerate(hash_functions_cm)])

    # Count-sketch final computation
    dict_count_sketch = exact_count.copy()

    for key in dict_count_sketch.keys():
        dict_count_sketch[key] = np.median([g_func(key) * cs_matrix[j, hash_func(key, W)] for j, (hash_func, g_func) in enumerate(hash_functions_cs)])

    topk_exact, topk_pred = topk_hitters(exact_count, dict_count_min, K)

    print("Port =", portExp, "T =", T, "D =", D, "W =", W, "K =", K)
    print("Number of processed items =", streamLength[0])
    print("Number of distinct items =", len(exact_count))
    print("Number of Top-K Heavy Hitters =", len(topk_exact))
    print("Avg Relative Error for Top-K Heavy Hitters with CM =", avg_error(exact_count, dict_count_min, K))
    print("Avg Relative Error for Top-K Heavy Hitters with CS =", avg_error(exact_count, dict_count_sketch, K))
    
    if K <= 10:
        topk_exact = dict(sorted(topk_exact.items()))
        print("Top-K Heavy Hitters:")
        for item in topk_exact.keys():
            print("Item", item, "True Frequency =", topk_exact[item], "Estimated Frequency with CM =", int(topk_pred[item]))