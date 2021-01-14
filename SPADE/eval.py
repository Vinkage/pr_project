import subprocess
import json
import skimage
import argparse
from pathlib import Path

from train import do_train
from test import do_test
# import test
import fid

# mock training and testing
# %run eval.py --name mock_gridsearch --dataset ../../mock_data

parser = argparse.ArgumentParser(description='Give name and dataset path')

parser.add_argument('--name', type=str,
                    help='an integer for the accumulator')
parser.add_argument('--dataset_path', type=str,
                    help='sum the integers (default: find the max)')

args = parser.parse_args() # %run eval.py
print(vars(args))

class grid_experiment:
    ## Stores information on paths in the project
    def __init__(
        self,
        name,
        dataset
    ):
        self.experiment_name = name
        self.dataset = dataset

    def name(self, search_key):
        return self.experiment_name + '_' + search_key

    def give_dataset_paths(self, train=False, test=False):
        dataset_root = Path(self.dataset)
        if train:
            images_path = dataset_root / 'train' / 'images'
            labels_path = dataset_root / 'train' / 'labels'
        if test:
            images_path = dataset_root / 'test' / 'images'
            labels_path = dataset_root / 'test' / 'labels'
        return {"image_dir": str(images_path),  "label_dir": str(labels_path)}


    ## Trains a model with a name and given train options based on the searches
    ## in the json
    def search(self, opt, train=False, test=False):
        # put the dictionary in the argparse format use by do_train
        arg_opt = argparse.Namespace()
        arg_opt_dict = vars(arg_opt)
        arg_opt_dict.update(opt)
        if train:
            do_train(arg_opt)
        if test:
            do_test(arg_opt)

    def print_options(self, options):
        pass

    def save_training_fid_stats(self):
        pass

    def scikit_metrics(self):
        pass


with open("train_defaults.json", "r") as d:
    train_defaults = json.load(d)

with open("test_defaults.json", "r") as d:
    test_defaults = json.load(d)

# loads dictionary with following structure
#
# {"search_name": {"option":"value(str/int)",...},
#    ...}
#
# This means that you have to input the grid manually in the searches.json file
#
# The search name is used to combine with the experiment name!
# This will be the name of you actual directory
with open("searches.json", "r") as g:
    searches = json.load(g)

# print(defaults)
# print(searches)

Grid = grid_experiment(args.name, args.dataset_path)

for search in searches.keys():
    # First of we name our search based on the commandline name given, and the
    # search name given in the json file you prepared.
    search_name = Grid.name(search)

    # build the dictionary of the stuff we want to give to do_train
    train_search = {**train_defaults,
                    **searches[search],
                    **{'name': search_name},
                    **Grid.give_dataset_paths(train=True)
                    }

    Grid.search(train_search, train=True)

    test_search = {**test_defaults,
                   **{'name': search_name},
                   **Grid.give_dataset_paths(test=True)
                   }

    Grid.search(test_search, test=True)

    # Call do_train


    # test_search = {**test_defaults,
    #                **searches[search],
    #                **{'name':Grid.name(searches[search])}
    #                }
