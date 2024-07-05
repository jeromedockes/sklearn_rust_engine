if __name__ == "__main__":
    import numpy as np
    from sklearn_rust_engine import lloyd_iter_chunked_dense
    X = np.random.randn(1000, 10)
    sample_weight = np.random.randn(1000)
    centers_old = np.random.randn(15, 10)
    centers_new = np.random.randn(15, 10)
    weight_in_clusters = np.random.randn(15)
    labels = np.random.randint(0,15, size=(1000,))
    centers_shift = np.random.randn(15,)
    n_threads = 4
    update_centers = True
    print(centers_new[0,0])
    lloyd_iter_chunked_dense(X, sample_weight, centers_old, centers_new, weight_in_clusters, labels, centers_shift, n_threads, update_centers)
    print(centers_new[0,0])
