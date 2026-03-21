import os
import matplotlib.pyplot as plt 
#plt.style.use('dark_background')
import numpy as np
import torch

def plot_test_sss(l_im,save_path,date,fig_lbl=None,cmap="BuGn",save=True,
                  fig_ext='.png', figs_defaults={'dpi':300, 'facecolor':'w', 'edgecolor':'w'}) :
    # 4 col: sss3, sss6, sss12 and sss12 true
    #fig,axes = plt.subplots(nrows=len(l_im),ncols=3,figsize=(35,15)) 
    fig,axes = plt.subplots(ncols=3, nrows=len(l_im), sharex=True, sharey=True,
                            figsize=(12,6*len(l_im)),
                            gridspec_kw={'hspace': 0.04,  'wspace': 0.10,
                                         'left':   0.04,  'right':  0.98,
                                         'top':    0.90,  'bottom': 0.04,
                                         })

    for n,line in enumerate(l_im):
        [sss6,sss12,target] = line
        diff = sss12 - target
        min_val = min(torch.min(sss6),torch.min(sss12),torch.min(target))
        max_val = max(torch.max(sss6),torch.max(sss12),torch.max(target))
        #im2 = axes[0].imshow(torch.squeeze(sss6).cpu().numpy(),cmap=cmap, vmin=min_val, vmax=max_val)
        im3 = axes[0].imshow(torch.squeeze(sss12).cpu().numpy(),cmap=cmap,  vmin=min_val, vmax=max_val)
        im4 = axes[1].imshow(torch.squeeze(target).cpu().numpy(),cmap=cmap,  vmin=min_val, vmax=max_val)
        im5 = axes[2].imshow(torch.squeeze(diff).cpu().numpy(),cmap="berlin")

        #fig.colorbar(im2,ax=axes[0], orientation='horizontal',label='pcu')
        fig.colorbar(im3,ax=axes[0], orientation='horizontal',label='pcu')
        fig.colorbar(im4,ax=axes[1], orientation='horizontal',label='pcu')
        fig.colorbar(im5,ax=axes[2], orientation='horizontal',label='pcu')
    
    cols=['pred sss12','true sss12', 'diff btwn out/valid']
    for ax, col in zip(axes, cols):
        ax.set_title(col)
    
    suptitle = f"Predicted and True SSS @ 1/12° on Test set"
    if fig_lbl is not None:
        suptitle += f" {fig_lbl}"
    suptitle += f" - [Training: {date}]"
    fig.suptitle(suptitle, y=0.98, size="large")
    
    #fig.tight_layout()
    
    if save:
        figs_file = f"{date}-SSS12-Pred-True-and-Diff"
        if fig_lbl is not None:
            figs_file += f"_Test-{fig_lbl.replace(' ','-').replace('(','').replace(')','')}"
        figs_file += "_images"
        figs_filename = os.path.join(save_path,figs_file+fig_ext)
        print("-- saving figure in file ... '{}'".format(figs_filename))
        plt.savefig(figs_filename, **figs_defaults)
    else:
        plt.show()

