import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re


def parse_loss_log_txt(loss_log):
    # Store parse results
    epochs_iterations_losses = {
        'epoch': [],
        'iters': [],
        'time': [],
        'GAN': [],
        'GAN_Feat': [],
        'VGG': [],
        'D_Fake': [],
        'D_real': []
        }
    # Regex to parse the per 100 iterations lines
    regex = "\(epoch:\s(?P<epoch>\d+).*iters:\s(?P<iters>\d+).*time:\s(?P<time>[\.\d]+).*GAN:\s(?P<GAN>[\.\d]+).*GAN_Feat:\s(?P<GAN_Feat>[\.\d]+).*VGG:\s(?P<VGG>[\.\d]+).*D_Fake:\s(?P<D_Fake>[\.\d]+).*D_real:\s(?P<D_real>[\.\d]+)"
    with open(loss_log) as l:
        lines = l.readlines()
        for line in lines:
            match = re.search(regex, line)
            if match:
                parse = match.groupdict()
                for key in parse.keys():
                    epochs_iterations_losses[key] += [ float(parse[key]) ]
    return epochs_iterations_losses

def discriminator_loss(loss_log_dict):
    sns.set_theme(style="darkgrid")
    df = pd.DataFrame(loss_log_dict)
    df['abs_iter'] = np.arange(len(df))
    df = pd.melt(df, id_vars='abs_iter', value_vars=['D_real', 'D_Fake'])
    line_plot = sns.lineplot(data=df, x="abs_iter", y="value", hue="variable")
    smooth_plot = sns.lmplot(data=df, x="abs_iter", y="value", hue="variable", scatter=False, logistic=True)
    plt.show()


loss_log_dict = parse_loss_log_txt("./loss_log.txt")

discriminator_loss(loss_log_dict)
