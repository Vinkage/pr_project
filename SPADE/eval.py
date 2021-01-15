# import ipdb
import subprocess
import json
import argparse
import torch
import numpy as np
import shutil
import tempfile
import os
import csv
from pathlib import Path

import skimage.transform
import skimage.io
import skimage.metrics

# import test
import fid
from inception_fid import InceptionV3

from train import do_train
from test import do_test

# mock training and testing
# %run eval.py --name mock_gridsearch --dataset ../mock_data

parser = argparse.ArgumentParser(description='Give name and dataset path')

parser.add_argument('--name', type=str,
                    help='an integer for the accumulator')
parser.add_argument('--dataset_path', type=str,
                    help='sum the integers (default: find the max)')
parser.add_argument('--root_path', type=str, default='.',
                    help='Root path of config files')

args = parser.parse_args() # %run eval.py
print(vars(args))
os.chdir(args.root_path)

def resize_images_and_save_statistics(grid_experiment = None, save_path = None):
    if grid_experiment is not None:
        training_images = grid_experiment.give_dataset_paths(train=True)
    else:
        raise "need to give grid experiment object at the moment"

    if isinstance(training_images, dict):
        imgs = training_images['image_dir']
    else:
        imgs = training_images

    # We need to resize the training images to 256x256 jpgs for the fid
    # calculation
    with tempfile.TemporaryDirectory() as tmp_dat:
        print('created temporary directory, resizing images and saving training FID stats', tmp_dat)

        # Resize loop which saves dataset in tmpdir
        image_list = os.listdir(imgs)
        i = 0
        for image in image_list:
            i += 1
            if i % 100 == 0:
                print(i, "out of ", len(image_list))
            
            # ipdb.set_trace()
            sk_img = skimage.io.imread(Path(imgs) / image)
            sk_img = skimage.transform.resize(sk_img,
                                    (256, 256),
                                    3,
                                    preserve_range=True,
                                    anti_aliasing=False)
            sk_img_filen = Path(tmp_dat) / image
            sk_img_filen_ext = sk_img_filen.with_suffix(".png")
            skimage.io.imsave(sk_img_filen_ext, sk_img.astype(np.uint8))

        ## Calcuate the fid statistics from the resi
        # Defaults from fid.py
        batch_size = 50
        dims = 2048

        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

        model = InceptionV3([block_idx]).to(device)

        m, s = fid.compute_statistics_of_path(tmp_dat, model, batch_size,
                                            dims, device)
        if grid_experiment is not None:
            np.savez(str(grid_experiment.exp_dir / 'FID_stats'), mu=m, sigma=s)


def fid_wrapper_grid(grid_experiment = None):
    if grid_experiment is not None:
        exp_dir = grid_experiment.exp_dir
    else:
        raise "need argument, either grid_experiment or manual path"

    train_fid_stats = exp_dir / 'FID_stats.npz'

    # Hard coded to use the latest epoch
    test_generated_image_path = Path('./results') / grid_experiment.current_search_name / 'test_latest' / 'images' / 'synthesized_image'

    batch_size = 50
    dims = 2048
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    # ipdb.set_trace()
    fid_value = fid.calculate_fid_given_paths([str(train_fid_stats), str(test_generated_image_path)],
                                              batch_size,
                                              device,
                                              dims)
    return fid_value

def resize_test_for_skimage_metrics(grid_experiment = None):
    if grid_experiment is not None:
        pass
    else:
        raise "need argument, either grid_experiment or manual path"
    # ipdb.set_trace()

    test_images_path = Path(grid_experiment.give_dataset_paths(test=True)['image_dir'])
    test_images_list = os.listdir(test_images_path)

    test_resized = grid_experiment.exp_dir / 'test_resized'
    test_resized.mkdir(exist_ok=True)

    for image in test_images_list:
        # ipdb.set_trace()
        sk_img = skimage.io.imread(Path(test_images_path) / image)
        sk_img = skimage.transform.resize(sk_img,
                                (256, 256),
                                3,
                                preserve_range=True,
                                anti_aliasing=False)
        sk_img_filen = test_resized / image
        sk_img_filen_ext = sk_img_filen.with_suffix(".png")
        skimage.io.imsave(sk_img_filen_ext, sk_img.astype(np.uint8))


def compute_ssim(real, fake):
    real = skimage.io.imread(real)/255
    fake = skimage.io.imread(fake)/255
    return skimage.metrics.structural_similarity(real, fake, multichannel=True, gaussian_weights=True, use_sample_covariance=False)

def compute_psnr(real, fake):
    real = skimage.io.imread(real)
    fake = skimage.io.imread(fake)
    return skimage.metrics.peak_signal_noise_ratio(real, fake)

def compute_rmse(real, fake):
    real = skimage.io.imread(real)
    fake = skimage.io.imread(fake)
    return skimage.metrics.normalized_root_mse(real, fake)

def skimage_metrics_grid(grid_experiment = None):
    if grid_experiment is not None:
        pass
    else:
        raise "need argument, either grid_experiment or manual path"

    # ipdb.set_trace()
    test_generated_image_path = Path('./results') / grid_experiment.current_search_name / 'test_latest' / 'images' / 'synthesized_image'
    test_true_image_path = grid_experiment.exp_dir / 'test_resized'

    generated_sorted_images = [test_generated_image_path / image for image in sorted(os.listdir(test_generated_image_path))]
    true_sorted_images = [test_true_image_path / image for image in sorted(os.listdir(test_true_image_path))]

    ssim = []
    psnr = []
    rmse = []
    for fake, real in zip(generated_sorted_images, true_sorted_images):
        fake, real = str(fake), str(real)
        # normalise the png values
        ssim += [compute_ssim(real, fake)]
        psnr += [compute_psnr(real, fake)]
        rmse += [compute_rmse(real, fake)]

    return {'ssim':np.mean(ssim), 'rmse':np.mean(rmse), 'psnr':np.mean(psnr)}

class grid_experiment():
    ## Stores information on paths in the project
    def __init__(
        self,
        name,
        dataset
    ):
        self.experiment_name = name
        self.dataset = dataset
        self._make_exp_dir()
        self.current_search_name = False
        self.current_metrics = False

    def _make_exp_dir(self):
        exp_path = Path('./grid_experiments/')
        if not exp_path.exists():
            exp_path.mkdir()
        exp_path = exp_path / self.experiment_name
        if not exp_path.exists():
            exp_path.mkdir()
        self.exp_dir = exp_path

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

    def dump_current_metrics(self):
        # ipdb.set_trace()
        if self.current_metrics == False:
            raise "no metrics to dump"

        metrics = self.current_metrics
        metrics.update({'search_name': self.current_search_name, 'experiment_name': self.experiment_name})

        csv_path = self.exp_dir / 'metric.csv'

        if not csv_path.exists():
            with open(csv_path, 'w') as f:
                f.write(', '.join(metrics.keys()) + '\n')
            with open(csv_path, 'a') as f:
                f.write(', '.join([str(value) for value in metrics.values()]) + '\n')
        elif csv_path.exists():
            with open(csv_path, 'a') as f:
                f.write(', '.join([str(value) for value in metrics.values()]) + '\n')



        self.current_metrics = False

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

Grid = grid_experiment(args.name, args.dataset_path)

# resize_images_and_save_statistics(grid_experiment = Grid)
# resize_test_for_skimage_metrics(grid_experiment = Grid)

# Check if the FID statistics were already calculated, you can re-use FID_stats
# from other grid experiments if they use the same dataset,
#
# but as you see then you need to copy them over to the experiment directory.
if not (Grid.exp_dir / 'FID_stats.npz').exists():
    resize_images_and_save_statistics(grid_experiment = Grid)
else:
    print("FID stats were calculated before.")

if not (Grid.exp_dir / 'test_resized').exists():
    resize_test_for_skimage_metrics(grid_experiment = Grid)
else:
    print("test images were resized before")

for search in searches.keys():
    # First of we name our search based on the commandline name given, and the
    # search name given in the json file you prepared.
    search_name = Grid.name(search)
    Grid.current_search_name = search_name
    Grid.current_metrics = {}

    # build the dictionary of the stuff we want to give to do_train
    train_search = {**train_defaults,
                    **searches[search],
                    **{'name': search_name},
                    **Grid.give_dataset_paths(train=True)
                    }

    # Training
    Grid.search(train_search, train=True)

    # build the stuff we give to do_test
    test_search = {**test_defaults,
                   **{'name': search_name},
                   **Grid.give_dataset_paths(test=True),
                   **{'norm_mode': train_search["norm_mode"]}
                   }

    # Inference
    Grid.search(test_search, test=True)

    # Add current fid value based on generated images and training fid statistics
    Grid.current_metrics['fid'] = fid_wrapper_grid(grid_experiment = Grid)

    # Add skimage metrics based on test images and generated images
    Grid.current_metrics.update(skimage_metrics_grid(grid_experiment = Grid))
    # ipdb.set_trace()

    Grid.dump_current_metrics()
