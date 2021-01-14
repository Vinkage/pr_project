"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict
import ipdb

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html

# %run test.py --name spade_log_test --load_size 256 --crop_size 256 --dataset_mode custom --label_dir ../../Celeb_subset/train/labels --image_dir ../../Celeb_subset/train/images  --label_nc 19 --no_instance --norm_mode spade

# Mock testing
# SPADE test
# %run test.py --name mock_spade --load_size 256 --crop_size 256 --dataset_mode custom --label_dir ../../mock_data/train/labels --image_dir ../../mock_data/train/images  --label_nc 19 --no_instance --norm_mode spade --which_epoch 1
# CLADE test
# %run test.py --name mock_clade --load_size 256 --crop_size 256 --dataset_mode custom --label_dir ../../mock_data/train/labels --image_dir ../../mock_data/train/images  --label_nc 19 --no_instance --norm_mode clade
# SEAN test
# %run test.py --name mock_sean --load_size 256 --crop_size 256 --dataset_mode custom --label_dir ../../mock_data/train/labels --image_dir ../../mock_data/train/images  --label_nc 19 --no_instance --norm_mode sean
ipdb.set_trace()

def do_test(opt):
    dataloader = data.create_dataloader(opt)

    model = Pix2PixModel(opt)
    model.eval()

    visualizer = Visualizer(opt)

    # create a webpage that summarizes the all results
    web_dir = os.path.join(opt.results_dir, opt.name,
                        '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir,
                        'Experiment = %s, Phase = %s, Epoch = %s' %
                        (opt.name, opt.phase, opt.which_epoch))

    # test
    for i, data_i in enumerate(dataloader):
        if i * opt.batchSize >= opt.how_many:
            break

        generated = model(data_i, mode='inference')

        img_path = data_i['path']
        for b in range(generated.shape[0]):
            print('process image... %s' % img_path[b])
            visuals = OrderedDict([('input_label', data_i['label'][b]),
                                ('synthesized_image', generated[b])])
            visualizer.save_images(webpage, visuals, img_path[b:b + 1])

    webpage.save()

if __name__=='__main__':
    options = TestOptions().parse()
    do_test(options)
