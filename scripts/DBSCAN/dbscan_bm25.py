import pickle
import re
import time
import pyterrier as pt
pt.init()

import argparse
import logging

import numpy as np
import pandas as pd
import hdbscan 

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
    cols = ['query', 'psg+', 'psg-']
    types = {col : str for col in cols}
    logging.info('Reading Text...')
    
    triples_df = pd.read_csv(args.textsource, sep='\t', header=None, index_col=False, names=cols, dtype=types)
    df = triples_df.copy()

    logging.info('Reading Embeddings...')
    with open(args.embedsource, 'rb') as f:
        array = np.load(f)

    per_cluster = args.candidates // args.nclust

    start = time.time()

    logging.info('Clustering Embeddings')
    clustering = hdbscan.HDBSCAN(min_cluster_size=per_cluster)
    c_idx = clustering.fit_predict(array)

    index = np.arange(len(array)).tolist()
    df['qid'] = ['q'+str(x) for x in index]
    df['cluster_id'] = c_idx.tolist()
    df['relative_index'] = index


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

    for i in range(args.nclust):
        tmp_df = df.loc[df['cluster_id']==i]
        scoring = scorer.score_set(tmp_df, 'psg+')
        scoring['diff'] = scoring['score'] - scorer.score_set(tmp_df, 'psg-')['score']
        tmp_idx = scorer.score_pairs(scoring)
    
        candidates = tmp_idx[:per_cluster]
        
        idx.extend(candidates)

    logging.info(f'{len(idx)} total candidates found')
    idx = np.random.choice(idx, args.candidates, replace=False)

    end = time.time()-start 
    logging.info(f'Completed Triples collection in {end} seconds')

    
    file = (idx, end)
    with open(args.out + f'clusterbm25.{args.nclust}.{args.candidates}.pkl', 'wb') as f:
        pickle.dump(file, f)

    return 0

if __name__ == '__main__':
    args = parser.parse_args()
    if args.verbose: log_level = logging.DEBUG
    else: log_level = logging.INFO
    logging.basicConfig(format='%(asctime)s - %(message)s', level=log_level)
    logging.info('--Initialising Candidate Choice Using Acceptance Threshold Sampler--')
    main(args)





