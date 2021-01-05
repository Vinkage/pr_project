- [Training command and details](#org03e5b76)
  - [Continue training](#org786f359)
  - [loss log plots](#orgece53a8)


<a id="org03e5b76"></a>

# Training command and details

The command I used was mostly adapted from the README file in the sean directory. The paths to the images and labels were relative paths to the 2000 training images and labels I randomly (seeded) sampled from the CelebA-HQ dataset. As with SEAN these were cropped to 256x256 during training.

```shell
python3 train.py --name mike_crop_subset --load_size 256 --crop_size 256 --dataset_mode custom --label_dir path/to/my/train/labels --image_dir path/to/my/train/images --label_nc 19 --no_instance --batchSize 2 --norm_mode clade
```

The difference with the SEAN command is that I enabled the CLADE norm<sub>mode</sub>, which is specific to CLADE without it SPADE resblks are used.

```python
# From CLADE/options/base_options.py
parser.add_argument('--norm_mode', type=str, default='spade', help='[spade | clade]')
```

More info on the training process is stored in text files in this directory.

-   [fid scores](fid.txt) only contains two lines. Need to find out when the fid is being calculated. The options file mentions only that the fid is calculated every 10 epochs.
-   [iter text](iter.txt) also only contains two lines. It is the 45 epochs times 2000 iterations the model has been trained for.
-   [loss log](loss_log.txt) contains the loss function value per epoch.


<a id="org786f359"></a>

## Continue training

I continued training using the following command:

```shell
python3 train.py --name mike_crop_subset --load_size 256 --crop_size 256 --dataset_mode custom --label_dir path/to/my/train/labels --image_dir path/to/my/train/images --label_nc 19 --no_instance --batchSize 8 --norm_mode clade --tf_log --serial_batches
```

I installed the following tenserflow version:

```shell
pip3 install tensorflow==1.15.0
```

After that for some reason I still had to replace in the visualisation utility `tf` to `tf.compat.v1`

```python
# visualiser.py

# search and replace
tf
# to
tf.compat.v1
```

When the training was interrupted the out event triggered and the stored iterations were written to a log file.


<a id="orgece53a8"></a>

## loss log plots

I used the tensorboard log files for this, It should also be able to export the relevant plots to publication quality images.
