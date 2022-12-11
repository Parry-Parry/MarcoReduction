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

class BM25scorer:
    def __init__(self, attr='text', index=None) -> None:
        self.attr = attr
        if index: self.scorer = pt.text.scorer(body_attr=attr, wmodel='BM25', background_index=index)
        else: self.scorer = pt.text.scorer(body_attr=attr, wmodel='BM25')

    def _convert_triple(self, df, focus):
        query_df = pd.DataFrame()
        query_df['qid'] = df['qid']
        query_df['query'] = df['query']
        query_df['docno'] = 'd1'
        query_df[self.attr] = df[focus]
        query_df['cluster_id'] = df['cluster_id']
        query_df['relative_index'] = df['relative_index']

        return query_df
    
    def score_set(self, df, focus):
        return self.scorer(self._convert_triple(df, focus))

    def score_pairs(self, df):
        scoring = df.sort_values(by=['diff'])['relative_index'].tolist()
        return scoring

def clean_text(text):
    return re.sub(r'[^A-Za-z0-9 ]+', '', text)

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
    
    triples_df = pd.read_csv(args.textsource, sep='\t', header=None, index_col=False, names=cols, dtype=types)
    df = triples_df.copy()

    logging.info('Reading Embeddings...')
    with open(args.embedsource, 'rb') as f:
        array = np.load(f)

    per_cluster = args.candidates // args.nclust

    config = ClusterConfig(
        niter=args.niter,
        nclust=args.nclust,
        cmin=per_cluster
    )

    start = time.time()

    logging.info('Clustering Embeddings')
    clustering = ClusterEngine(config)
    clustering.train(array)
    c_idx = clustering.query(array)
    index = np.arange(len(array)).tolist()
    df['qid'] = ['q'+str(x) for x in index]
    df['cluster_id'] = c_idx.tolist()
    df['relative_index'] = index

    counts = df['cluster_id'].value_counts()

    scale = counts.median()
    diff = floor(scale / per_cluster)
    print(f'Adding {diff} extra samples when over median: {scale} and max {counts.max()}')

    logging.info('Cleaning Text...')
    df['query'] = df['query'].apply(clean_text)
    df['psg+'] = df['psg+'].apply(clean_text)
    df['psg-'] = df['psg-'].apply(clean_text)
    
    idx =[]
    logging.info('In Centroid Ranking with BM25...')
   
    if args.index: 
        ds = pt.get_dataset(args.index)
        index = pt.IndexFactory.of(ds.get_index(variant='terrier_stemmed'))
    else: index = None 
    scorer = BM25scorer(index=index)

    scored = scorer.score_set(df, 'psg+')
    neg = scorer.score_set(df, 'psg-')
    scored['diff'] = scored['score'] - neg['scored']

    for i in range(args.nclust):
        tmp_df = scored.loc[scored['cluster_id']==i]
        scores = scorer.score_pairs(tmp_df, per_cluster)
        if len(scores) >= per_cluster:
            if counts[i] > scale: candidates = scores[:per_cluster+diff]
            else: candidates = scores[:per_cluster]
        else:
            logging.info(f'Cluster {i} has too few candidates: {len(id)} found')
            candidates = scores
        idx.extend(candidates)

    logging.info('Retrieving Relevant IDs')
    new_df = triples_df.loc[idx]

    end = time.time()-start 
    logging.info(f'Completed Triples collection in {end} seconds')

    new_df.to_csv(args.out, sep='\t', header=False, index=False)
    return 0

if __name__ == '__main__':
    args = parser.parse_args()
    if args.verbose: log_level = logging.DEBUG
    else: log_level = logging.INFO
    logging.basicConfig(format='%(asctime)s - %(message)s', level=log_level)
    logging.info('--Initialising Candidate Choice Using Acceptance Threshold Sampler--')
    main(args)





