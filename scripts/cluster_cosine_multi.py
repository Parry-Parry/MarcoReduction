import re
import time
import pyterrier as pt
pt.init()

import argparse
import logging
from typing import NamedTuple
import multiprocessing as mp
from math import ceil, floor

import numpy as np
import pandas as pd
import faiss 

from scipy.spatial.distance import cosine

class ClusterConfig(NamedTuple):
    niter : int
    nclust : int 
    cmin : int 

class ClusterEngine:
    def __init__(self, config) -> None:
        self.niter = config.niter
        self.nclust = config.nclust
        self.min = config.cmin 
        self.kmeans = None
    
    def query(self, x) -> np.array:
        assert self.kmeans is not None
        _, I = self.kmeans.index.search(x, 1)
        return I.ravel()

    def train(self, x) -> None:
        self.kmeans = faiss.Kmeans(x.shape[-1], self.nclust, niter=self.niter, verbose=False, spherical=True, min_points_per_centroid=self.min)
        self.kmeans.train(x)

def cosine_scoring(array):
    q = array[:args.dim]
    p1 = array[args.dim:2*args.dim]
    p2 = array[2*args.dim:3*args.dim]

    return cosine(q, p1) - cosine(q, p2)

parser = argparse.ArgumentParser()

parser.add_argument('-textsource', type=str)
parser.add_argument('-embedsource', type=str)
parser.add_argument('-niter', type=int)
parser.add_argument('-nclust', type=int, nargs='+')
parser.add_argument('-dim', type=int)
parser.add_argument('-candidates', type=int)
parser.add_argument('-out', type=str)

parser.add_argument('--index', type=str)
parser.add_argument('--verbose', action='store_true')

def main(args):
    faiss.omp_set_num_threads(mp.cpu_count())

    cols = ['query', 'psg+', 'psg-']
    types = {col : str for col in cols}
    logging.info('Reading Text...')
    
    triples_df = pd.read_csv(args.textsource, sep='\t', header=None, index_col=False, names=cols, dtype=types)

    logging.info('Reading Embeddings...')
    with open(args.embedsource, 'rb') as f:
        array = np.load(f)

    for c in args.nclust:

        per_cluster = args.candidates // c

        config = ClusterConfig(
            niter=args.niter,
            nclust=c,
            cmin=per_cluster
        )

        start = time.time()

        logging.info('Clustering Embeddings')
        clustering = ClusterEngine(config)
        clustering.train(array)
        c_idx = clustering.query(array)
        index = np.arange(len(array))

        counts = np.unique(c_idx)

        scale = np.median(counts)
        diff = floor(scale / per_cluster)
        print(f'Adding {diff} extra samples when over median: {scale} and max {counts.max()}')
        
        idx =[]
        logging.info('In Centroid Ranking with Cosine Similarity...')

        for i in range(c):
            tmp_array = array[np.where(c_idx==i)]
            tmp_idx = index[np.where(c_idx==i)]
            scoring = np.apply_along_axis(cosine_scoring, 1, tmp_array)
            ranked_idx = tmp_idx[np.argsort(scoring)]
            if len(tmp_array) >= per_cluster:
                if counts[i] > scale: candidates = ranked_idx[:per_cluster+diff].tolist()
                else: candidates = ranked_idx[:per_cluster].tolist()
            else:
                logging.info(f'Cluster {i} has too few candidates: {len(ranked_idx)} found')
                candidates = ranked_idx.tolist()
            idx.extend(candidates)

        logging.info(f'{len(idx)} total candidates found')
        idx = np.random.choice(idx, args.candidates, replace=False)

        logging.info('Retrieving Relevant IDs')
        new_df = triples_df.loc[idx]

        end = time.time()-start 
        logging.info(f'Completed Triples collection in {end} seconds')

        new_df.to_csv(args.out + f'cosine.{c}.tsv', sep='\t', header=False, index=False)
        return 0

if __name__ == '__main__':
    args = parser.parse_args()
    if args.verbose: log_level = logging.DEBUG
    else: log_level = logging.INFO
    logging.basicConfig(format='%(asctime)s - %(message)s', level=log_level)
    logging.info('--Initialising Candidate Choice Using Cosine Ranking Sampler--')
    main(args)





