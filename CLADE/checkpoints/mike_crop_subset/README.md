- [Training command and details](#orgd9e89e1)
- [Adverserial loss term](#org88f3504)
  - [Hinge version](#org00f83e4)
  - [SPADE implementation](#org12be6c7)


<a id="orgd9e89e1"></a>

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


<a id="org88f3504"></a>

# Adverserial loss term

The loss function is the same as with the pix2pixHD paper, instead they use a hinge loss form for the generator loss.

The general GAN loss function in pix2pixHD:

\[ \min_{G} \max_{D} \mathcal{L}(G, D) \]

we use a multiscale discriminator by default, which you can check in the multiscale discriminator class in SPADE.

```python
# From models/networks/discriminator.py
class MultiscaleDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--netD_subarch', type=str, default='n_layer',
                            help='architecture of each discriminator')
        parser.add_argument('--num_D', type=int, default=2,
                            help='number of discriminators to be used in multiscale')
        opt, _ = parser.parse_known_args()

        # define properties of each discriminator of the multiscale discriminator
        subnetD = util.find_class_in_module(opt.netD_subarch + 'discriminator',
                                            'models.networks.discriminator')
        subnetD.modify_commandline_options(parser, is_train)

        return parser
# [...]
```

So the general loss form is actually (also in SEAN/SPADE),

\[ \min_{E,G} \max_{D_1 , D_2} \sum_{k=1,2}^{} \mathcal{ L }_{GAN}(E,G, D_{k}) \]

which is the minimax game objective. The objective function \( \mathcal{L}\) is given by,

\[ \mathcal{L}_{GAN}(G,D) = \mathbb{E}_{\left( \boldsymbol{s,x}\right)} \left[ \log D(\boldsymbol{s,x}) \right] + \mathbb{E}_{\boldsymbol{s}} \left[ \log(1 - D(\boldsymbol{s} , G(\boldsymbol{s}))) \right] \]

Where \( s\) is the label map, and \( x\) is the image.

Note that \( D\) is a (set of) fully convolutional network(s) with a sigmoidal activation function at the end (only in pix2pix paper, or original gan<sub>mode</sub> loss in SPADE project). This means that the range of \( D\) should be \( \left[ 0,1\right] \). Where *one* means real and *zero* means fake.


<a id="org00f83e4"></a>

## Hinge version

Now in later papers (SPADE and its derivatives) a hinge form was used, without any sigmoid predictions. This is best explained in the SEAN paper,

\begin{align}
\mathcal{ L }_{GAN} &= \mathbb{E}_{} \left[ \max(0,1 - D_{k}(\boldsymbol{s,x})) \right] \tag*{\{\} }\\
 &+ \mathbb{E}_{} \left[ \max(0, 1 + D_{k}(\boldsymbol{s,},G(\boldsymbol{s}))) \right] \tag*{\{\} }
\end{align}

Where again \( s\) is the label map and \( x\) is the real image. You can see that there are two hinge terms, the real and fake discriminator loss.

This is equivalent to the following (Zhang et al. 2019: SAGAN):

\begin{align}
\mathcal{ L }_{D} &= - \mathbb{E}_{(\boldsymbol{s,x})} \left[ \min(0, -1 + D(\boldsymbol{s,x})) \right] \tag*{\{\} }\\
 &- \mathbb{E}_{s} \left[ \min(0, -1 - D(G(\boldsymbol{s}) , \boldsymbol{s})) \right] \tag*{\{\} }\\
\mathcal{ L }_{G} &= - \mathbb{E}_{\boldsymbol{s} } \left[ D(G(\boldsymbol{s}) , \boldsymbol{s}) \right] \tag*{\{\} }
\end{align}

Where \( \mathcal{ L }_{G}\) is the generator loss, this is important, because we are training stepwise the generator and discriminator. One step the \(\mathcal{ L }_{D}\) is computed and \( \mathcal{ L }_{G}\) in the other.

It can be shown that this equation converges to \(2 \) , and that is equivalent to pushing the generated image to the separating hyperplane, and optimising the hyperplane margins for the discriminator (geometric gan paper).

The intuition for this is that when the probability distribution of the real images and fake images are equivalent, or the reverse KL-divergence \( KL \left[ p_{g} || q_{data}\right]\) is minimised (<sup id="e7b68df19302656fb2b29d281c39ec13"><a href="#miyatoSpectralNormalizationGenerative2018" title="Miyato, Kataoka, Koyama \&amp; Yoshida, Spectral {{Normalization}} for {{Generative Adversarial Networks}}, {arXiv:1802.05957 [cs, stat]}, v(), (2018).">miyatoSpectralNormalizationGenerative2018</a></sup>)


<a id="org12be6c7"></a>

## SPADE implementation

The actual output of a forward pass through the MultiScaleDiscriminator is a nested list of dimension, n<sub>D</sub> x n<sub>Layers</sub><sub>in</sub><sub>D</sub>. The discriminator is actually the two times application of a the normal Nlayer patchgan discriminator. It is a four layer fully convolutional network. The first time the input is normal, the second time the input is downsampled.

This nested list is used in the discriminate method of the pix2pix model class.

This tensor is fed to the divide<sub>pred</sub> method of pix2pix to give the predictions to the loss class.
