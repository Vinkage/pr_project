import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from CLADE.fid_score import calculate_fid_given_paths

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

for x in os.walk("./results"):
    fid_value = calculate_fid_given_paths(x,
                                          args.batch_size,
                                          args.gpu != '',
                                          args.dims,
                                          args)

    print("model ", x, 'has FID: ', fid_value)

