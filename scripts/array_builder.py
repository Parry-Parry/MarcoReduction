from typing import Any, List
import numpy as np
import pandas as pd
from pandas.io.parsers import TextFileReader
from sentence_transformers import SentenceTransformer
import argparse
import logging 
from queryreduce.models.vectorfactory import VectorFactory

'''
Retrieve text associated with triplets of IDs, generate embeddings and return tuples
'''

parser = argparse.ArgumentParser(description='Construct embedding clusters from triplets of IDs')

parser.add_argument('-dataset', '-d', type=str, help='Directory of Triples tsv')
parser.add_argument('-model', type=str, help='Name of Model Checkpoint for indexing [Sentence Transformer Compatible]')
parser.add_argument('-batch_size', type=int, help='Batch Size')
parser.add_argument('-out', type=str, help='Output file [NPY]')


def main(args):
    cols = ['qid', 'pid+', 'pid-']
    types = {col : str for col in cols}
    logging.info('Reading Dataset...')
    
    iterator_df = pd.read_csv(args.dataset, sep='\t', header=None, index_col=False, names=cols, dtype=types, chunksize=args.batch_size)

    logging.info(f'Running Vector Factory on model {args.model} with batch size {args.batch_size}')
    factory = VectorFactory(args.model)
    factory.run(iterator_df, args.out)
    logging.info('Completed Successfully')

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    main(parser.parse_args())