import argparse
from math import trunc 
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='Generate subsets of any array')

parser.add_argument('-source', type=str, default=None, help='Dataset Source')
parser.add_argument('-portion', type=int, help=r'% of dataset to use')
parser.add_argument('-dupes', type=int, help='How many random subsets')
parser.add_argument('-suffix', type=str, default='train', help='Suffix of output name')
parser.add_argument('-out', type=str, help='Output dir for tsv')

def main(args):
    cols = ['qid', 'pid+', 'pid-']
    types = {col : str for col in cols}
   
    df = pd.read_csv(args.source, sep='\t', header=None, index_col=False, names=cols, dtype=types)
    num_choices = trunc(len(df) * (args.portion/100))

    for i in range(args.dupes):
        sub_df = df.sample(n=num_choices)
        sub_df.to_csv(args.out + f'triples.{args.suffix}.{i}.tsv', sep='\t', index=False, header=False)

if __name__ == '__main__':
    main(parser.parse_args())