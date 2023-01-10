import argparse
import logging
import os
import pickle 
import pandas as pd 
import subprocess

parser = argparse.ArgumentParser()

parser.add_argument('-source', type=str)
parser.add_argument('-sink', type=str)
parser.add_argument('-text', type=str)
parser.add_argument('-maxsteps', type=int)
parser.add_argument('-root', type=str)
parser.add_argument('-experiment', type=str)

parser.add_argument('--prefix', type=str)
parser.add_argument('--verbose', action='store_true')

def main(args):
    files = [f for f in os.listdir(args.source) if os.path.isfile(os.path.join(args.source, f))]

    cols = ['query', 'psg+', 'psg-']
    types = {col : str for col in cols}
    
    df = pd.read_csv(args.text, sep='\t', header=None, index_col=False, names=cols, dtype=types)

    for file in files:    
        strip_name = file.strip('.pkl')
        if args.prefix:
            strip_name = f'{args.prefix}.' + strip_name

        logging.info(f'Training ColBERT for run {strip_name}')
        with open(os.path.join(args.source, file), 'rb') as f:
            idx, _, _ = pickle.load(f)

        tmp = df.loc[idx]
        tsv_name = f"{strip_name}.tsv"
        tmp.to_csv(os.path.join(args.sink, tsv_name), sep='\t', header=False, index=False)
        del tmp 

        out = subprocess.run([
            'python', '-m', 'colbert.train', '--amp', '--doc_maxlen', '180', '--mask-punctuation', '--maxsteps', str(args.maxsteps), '--triples', os.path.join(args.sink, tsv_name), '--root', args.root, '--experiment', args.experiment, '--similarity', 'l2', '--run', strip_name
        ])
        
        
if __name__ == '__main__':
    args = parser.parse_args()
    if args.verbose: log_level = logging.DEBUG
    else: log_level = logging.INFO
    logging.basicConfig(format='%(asctime)s - %(message)s', level=log_level)
    logging.info('--Running Parameter Sweep for ColBERT--')
    main(args)
    
    

        