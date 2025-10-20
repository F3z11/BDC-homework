import numpy as np
import pandas as pd
import sys

def generate_dataset(N, K):
    np.random.seed(100)

    points_per_cluster = N // K
    top_n = int(points_per_cluster * 0.9)
    bottom_n = points_per_cluster - top_n

    data = []

    for i in range(K):
        x_offset = i * 200 # parameter to control the distance between clusters

        # Larger cluster (label A)
        top_x = np.random.normal(loc=20, scale=15, size=top_n) + x_offset
        top_y = np.random.normal(loc=250, scale=10, size=top_n)
        top_points = zip(top_x, top_y)
        data.extend([(x, y, 'A') for x, y in top_points])

        # Smaller cluster (label B)
        bottom_x = np.random.normal(loc=0, scale=8, size=bottom_n) + x_offset
        bottom_y = np.random.normal(loc=20, scale=7, size=bottom_n)
        bottom_points = zip(bottom_x, bottom_y)
        data.extend([(x, y, 'B') for x, y in bottom_points])

    for row in data:
        print(f"{row[0]},{row[1]},{row[2]}")

def main():
    assert len(sys.argv) == 3, "Usage: python G22GEN.py <N> <K>"

    N = sys.argv[1] #num of points
    assert N.isdigit(), "N must be an integer"
    N = int(N)

    K = sys.argv[2] #num of centroids
    assert K.isdigit(), "K must be an integer"
    K = int(K)

    generate_dataset(N=N, K=K)

if __name__ == "__main__":
    main()