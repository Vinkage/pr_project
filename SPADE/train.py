"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
import ipdb
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer

# CLADE training
# %run train.py --name ipdb_test --load_size 256 --crop_size 256 --dataset_mode custom --label_dir ../../Celeb_subset/train/labels --image_dir ../../Celeb_subset/train/images  --label_nc 19 --no_instance --norm_mode clade
#
# SEAN training
# %run train.py --name ipdb_test --load_size 256 --crop_size 256 --dataset_mode custom --label_dir ../../Celeb_subset/train/labels --image_dir ../../Celeb_subset/train/images  --label_nc 19 --no_instance --norm_mode sean --save_epoch_freq 1
# ipdb.set_trace() # my breakpoint

# Mock testing
# SPADE train
# %run train.py --name mock_spade --load_size 256 --crop_size 256 --dataset_mode custom --label_dir ../../mock_data/train/labels --image_dir ../../mock_data/train/images  --label_nc 19 --no_instance --norm_mode spade --save_epoch_freq 1 --niter 1 --display_freq 1
# CLADE train
# %run train.py --name mock_clade --load_size 256 --crop_size 256 --dataset_mode custom --label_dir ../../mock_data/train/labels --image_dir ../../mock_data/train/images  --label_nc 19 --no_instance --norm_mode clade --save_epoch_freq 1 --niter 1
# SEAN train
# %run train.py --name mock_sean --load_size 256 --crop_size 256 --dataset_mode custom --label_dir ../../mock_data/train/labels --image_dir ../../mock_data/train/images  --label_nc 19 --no_instance --norm_mode sean --save_epoch_freq 1 --niter 1

# parse options
ipdb.set_trace()
# opt = TrainOptions().parse()

# print options to help debugging
# print(' '.join(sys.argv))
# train.py --name ipdb_test --load_size 256 --crop_size 256 --dataset_mode custom --label_dir ../../Celeb_subset/train/labels --image_dir ../../Celeb_subset/train/images --label_nc 19 --no_instance

# load the dataset
def do_train(opt):
    dataloader = data.create_dataloader(opt)
    # dataset [CustomDataset] of size 2000 was created

    # create trainer for our model
    trainer = Pix2PixTrainer(opt)
    # Network [SPADEGenerator] was created. Total number of parameters: 92.5 million. To see the architecture, do print(network).
    # Network [MultiscaleDiscriminator] was created. Total number of parameters: 5.6 million. To see the architecture, do print(network).
    # Downloading: "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth" to /root/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth
    # HBox(children=(FloatProgress(value=0.0, max=574673361.0), HTML(value='')))

    # create tool for counting iterations
    iter_counter = IterationCounter(opt, len(dataloader))

    # create tool for visualization
    visualizer = Visualizer(opt)
    # create web directory ./checkpoints/ipdb_test/web...

    for epoch in iter_counter.training_epochs():
        iter_counter.record_epoch_start(epoch)
        for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
    # data_i =
    # {'label': tensor([[[[ 0.,  0.,  0.,  ...,  0.,  0.,  0.],
    #           [ 0.,  0.,  0.,  ...,  0.,  0.,  0.],
    #           [ 0.,  0.,  0.,  ...,  0.,  0.,  0.],
    #           ...,
    #           [ 0.,  0.,  0.,  ..., 13., 13., 13.],
    #           [ 0.,  0.,  0.,  ..., 13., 13., 13.],
    #           [ 0.,  0.,  0.,  ..., 13., 13., 13.]]]]), 'instance': tensor([0]), 'image': tensor([[[[-1.0000, -1.0000, -0.9922,  ...,  0.5529,  0.5529,  0.5529],
    #           [-1.0000, -1.0000, -0.9922,  ...,  0.5529,  0.5529,  0.5529],
    #           [-1.0000, -0.9922, -0.9843,  ...,  0.5529,  0.5529,  0.5529],
    #           ...,
    #           [ 0.4118,  0.4275,  0.4118,  ..., -0.7490, -0.7333, -0.7020],
    #           [ 0.4196,  0.4039,  0.4196,  ..., -0.7020, -0.7804, -0.7255],
    #           [ 0.4039,  0.4196,  0.4588,  ..., -0.6784, -0.7333, -0.6941]],

    #          [[-0.9529, -0.9686, -0.9843,  ...,  0.5843,  0.5843,  0.5843],
    #           [-0.9529, -0.9686, -0.9843,  ...,  0.5843,  0.5843,  0.5843],
    #           [-0.9608, -0.9686, -0.9765,  ...,  0.5843,  0.5843,  0.5843],
    #           ...,
    #           [ 0.4431,  0.4588,  0.4431,  ..., -0.8510, -0.8353, -0.8039],
    #           [ 0.4510,  0.4353,  0.4510,  ..., -0.8039, -0.8824, -0.8275],
    #           [ 0.4353,  0.4510,  0.4902,  ..., -0.7725, -0.8275, -0.7882]],

    #          [[-0.9843, -1.0000, -1.0000,  ...,  0.6549,  0.6549,  0.6549],
    #           [-0.9843, -1.0000, -1.0000,  ...,  0.6549,  0.6549,  0.6549],
    #           [-0.9922, -1.0000, -0.9922,  ...,  0.6549,  0.6549,  0.6549],
    #           ...,
    #           [ 0.5294,  0.5451,  0.5294,  ..., -0.9216, -0.8980, -0.8667],
    #           [ 0.5373,  0.5216,  0.5373,  ..., -0.8824, -0.9529, -0.8980],
    #           [ 0.5216,  0.5373,  0.5765,  ..., -0.8667, -0.9216, -0.8824]]]]), 'path': ['../../Celeb_subset/train/images/8516.jpg']}
            iter_counter.record_one_iteration()

            # Training
            # train generator
            if i % opt.D_steps_per_G == 0:
                trainer.run_generator_one_step(data_i)

            # train discriminator
            trainer.run_discriminator_one_step(data_i)

            # Visualizations
            if iter_counter.needs_printing():
                losses = trainer.get_latest_losses()
                visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                                losses, iter_counter.time_per_iter)
                visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

            if iter_counter.needs_displaying():
                visuals = OrderedDict([('input_label', data_i['label']),
                                    ('synthesized_image', trainer.get_latest_generated()),
                                    ('real_image', data_i['image'])])
                visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

            if iter_counter.needs_saving():
                print('saving the latest model (epoch %d, total_steps %d)' %
                    (epoch, iter_counter.total_steps_so_far))
                trainer.save('latest')
                iter_counter.record_current_iter()

        trainer.update_learning_rate(epoch)
        iter_counter.record_epoch_end()

        if epoch % opt.save_epoch_freq == 0 or \
        epoch == iter_counter.total_epochs:
            print('saving the model at the end of epoch %d, iters %d' %
                (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            trainer.save(epoch)

    print('Training was successfully finished.')

if __name__ == '__main__':
  options = TrainOptions().parse()
  do_train(options)
