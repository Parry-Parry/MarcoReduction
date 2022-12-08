import pyterrier as pt
pt.init()

import argparse
import logging
from typing import NamedTuple
import multiprocessing as mp
from math import ceil

import numpy as np
import pandas as pd
import faiss 

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
        _, I = self.kmeans.search(x, 1)
        return I.ravel()

    def train(self, x) -> None:
        self.kmeans = faiss.Kmeans(x.shape[-1], self.nclust, niter=self.niter, verbose=False, spherical=True, min_points_per_centroid=self.min)
        self.kmeans.train(x)

class BM25scorer:
    def __init__(self, attr='text', index=None) -> None:
        self.attr = attr
        if index: self.scorer = pt.text.scorer(body_attr=attr, wmodel='BM25', background_index=index)
        else: self.scorer = pt.text.scorer(body_attr=attr, wmodel='BM25')

    def _convert_triple(self, df):
        query_df = pd.DataFrame()
        query_df['qid'] = df['qid']
        query_df['query'] = df['query']
        query_df['docno'] = 'd1'
        query_df[self.attr] = df['psg+']
        query_df['cluster_id'] = df['cluster_id']
        query_df['relative_index'] = df['relative_index']

        return query_df
    
    def score_set(self, df):
        return self.scorer(self._convert_triple(df))

    def score_pairs(self, df, n):
        assert len(df) >= n
        scoring = df.sort_values(by=['score'])['relative_index'].to_list()
        return scoring[:n]

parser = argparse.ArgumentParser()

parser.add_argument('-textsource', type=str)
parser.add_argument('-embedsource', type=str)
parser.add_argument('-niter', type=int)
parser.add_argument('-nclust', type=int)
parser.add_argument('-candidates', type=int)
parser.add_argument('-out', type=str)

parser.add_argument('--index', type=str)
parser.add_argument('--verbose', action='store_true')

def main(args):
    faiss.omp_set_num_threads(mp.cpu_count())
    cols = ['query', 'psg+', 'psg-']
    types = {col : str for col in cols}
    logging.info('Reading Text...')
    
    df = pd.read_csv(args.dataset, sep='\t', header=None, index_col=False, names=cols, dtype=types)

    logging.info('Reading Embeddings...')
    with open(args.embedsource, 'rb') as f:
        array = np.load(f)

    per_cluster = ceil(args.candidates / args.nclust)

    config = ClusterConfig(
        niter=args.niter,
        nclust=args.nclust,
        cmin=per_cluster
    )

    logging.info('Clustering Embeddings')
    clustering = ClusterEngine(config)
    clustering.train(array)
    c_idx = clustering.query(array)
    index = np.arange(len(array)).to_list()
    df['qid'] = ['q'+str(x) for x in index]
    df['cluster_id'] = c_idx.to_list()
    df['relative_index'] = index
    
    idx =[]
    logging.info('In Centroid Ranking with BM25...')
    if args.index: index = args.index
    else: index = None 
    scorer = BM25scorer(index=index)

    scored = scorer.score_set(df)

    for i in range(args.nclust):
        tmp_df = scored.loc[scored['cluster_id']==i]
        idx.update(scorer.score_pairs(tmp_df, per_cluster))

    logging.info('Retrieving Relevant IDs')
    new_df = df.loc[idx]

    new_df.to_csv(args.out, sep='\t', header=False, index=False)
    return 0

if __name__ == '__main__':
    args = parser.parse_args()
    if args.verbose: log_level = logging.DEBUG
    else: log_level = logging.INFO
    logging.basicConfig(format='%(asctime)s - %(message)s', level=log_level)
    logging.info('--Initialising Candidate Choice Using Acceptance Threshold Sampler--')
    main(args)




