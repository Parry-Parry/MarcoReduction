import time
import pyterrier as pt
pt.init()

import argparse
import logging
from typing import NamedTuple
import multiprocessing as mp
import re 

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

    def _convert_triple(self, df):
        query_df = pd.DataFrame()
        query_df['qid'] = df['qid']
        query_df['query'] = df['query']
        query_df['docno'] = 'd1'
        query_df[self.attr] = df['psg+']
        query_df['relative_index'] = df['relative_index']

        return query_df

    def score_pairs(self, df):
        score_obj = self.scorer.transform(self._convert_triple(df))
        scoring = score_obj.sort_values(by=['score'])['relative_index'].tolist()
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
    
    df = pd.read_csv(args.textsource, sep='\t', header=None, index_col=False, names=cols, dtype=types)


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

    logging.info('Clustering Embeddings...')
    clustering = ClusterEngine(config)
    clustering.train(array)
    c_idx = clustering.query(array)
    index = np.arange(len(array)).tolist()
    df['qid'] = ['q'+str(x) for x in index]
    df['cluster_id'] = c_idx.tolist()
    df['relative_index'] = index

    logging.info('Cleaning Text...')
    df['query'] = df['query'].apply(clean_text)
    df['psg+'] = df['psg+'].apply(clean_text)
    df['psg-'] = df['psg-'].apply(clean_text)

    idx =[]
    logging.info(f'In Centroid Ranking with BM25 with {per_cluster} candidates per cluster...')
    if args.index: index = pt.BatchRetrieve.from_dataset('msmarco_passage', args.index, wmodel='BM25')
    else: index = None 
    scorer = BM25scorer(index=index)
    for i in range(args.nclust):
        tmp_df = df.loc[df['cluster_id']==i]
        id = scorer.score_pairs(tmp_df)
        if len(id) >= per_cluster:
            candidates = id[:per_cluster]
        else:
            logging.info(f'Cluster {i} has too few candidates: {len(id)} found')
            candidates = id
        idx.extend(candidates)

    logging.info('Retrieving Relevant IDs')
    new_df = df.loc[idx]

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





