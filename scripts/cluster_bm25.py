import numpy as np
import faiss 

class clusterEngine:
    def __init__(self, config) -> None:
        self.niter = config.niter
        self.nclust = config.nclust
        self.min = config.min 
        self.kmeans = None
    
    def query(self, x) -> np.array:
        assert self.kmeans is not None
        _, I = self.kmeans.search(x, 1)
        return I.ravel()

    def train(self, x) -> None:
        self.kmeans = faiss.Kmeans(x.shape[-1], self.nclust, niter=self.niter, verbose=False, spherical=True, min_points_per_centroid=self.min)
        self.kmeans.train(x)