import numpy as np


def calc_euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


def calc_weight(x, kernel='flat'):
    if x <= 1:
        if kernel.lower() == 'flat':
            return 1
        elif kernel.lower() == 'gaussian':
            return np.exp(-1 * (x ** 2))
        else:
            raise Exception("'%s' is invalid kernel" % kernel)
    else:
        return 0


def mean_shift(X, bandwidth, n_iteration=20, epsilon=0.001):
    centroids = np.zeros_like(X)
    kernel = 'gaussian'
    for i in range(len(X)):
        centroid = X[i].copy()  # 초기 중심점(t_0) 설정 -> 각 datapoint를 초기 중심점으로 할당
        prev = centroid.copy()
        t = 0
        while True:
            dists = np.array([calc_euclidean_distance(centroid, p) for p in X])  # 중심점과 모든 점들과 거리 계산
            weighted_sum = np.array([p * calc_weight(dists[idx] / bandwidth, kernel=kernel) \
                                     for idx, p in enumerate(X)])  #
            weighted_sum = np.sum(weighted_sum, axis=0)

            re_term = sum([calc_weight(dists[idx] / bandwidth, kernel=kernel) \
                           for idx, p in enumerate(X)])

            centroid = weighted_sum / re_term
            if t > n_iteration or calc_euclidean_distance(centroid, prev) <= epsilon:
                break
            prev = centroid.copy()
            t += 1
        centroids[i] = centroid.copy()
    return centroids


def mean_shift_with_history(X, bandwidth, n_iteration=20, epsilon=0.001):
    history = {}
    kernel = 'gaussian'
    for i in range(len(X)):
        history[i] = []
    centroids = np.zeros_like(X)

    for i in range(len(X)):
        centroid = X[i].copy()  # 초기 중심점(t_0) 설정 -> 각 datapoint를 초기 중심점으로 할당
        prev = centroid.copy()
        history[i].append(centroid.copy())

        t = 0
        while True:
            dists = np.array([calc_euclidean_distance(centroid, p) for p in X])  # 중심점과 모든 점들과 거리 계산
            weighted_sum = np.array([p * calc_weight(dists[idx] / bandwidth, kernel=kernel) \
                                     for idx, p in enumerate(X)])  #
            weighted_sum = np.sum(weighted_sum, axis=0)

            re_term = sum([calc_weight(dists[idx] / bandwidth, kernel=kernel) \
                           for idx, p in enumerate(X)])

            centroid = weighted_sum / re_term
            if t > n_iteration or calc_euclidean_distance(centroid, prev) <= epsilon:
                break
            history[i].append(centroid.copy())
            prev = centroid.copy()
            t += 1

        centroids[i] = centroid.copy()

    return centroids, history
