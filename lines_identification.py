#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
import pandas as pd
from astropy import units as u
abunds_idents={"h":1,"he":2,"li":3,"be":4,"b":5,"c":6,"n":7,"o":8,"f":9,"ne":10,"na":11,"mg":12,"al":13,"si":14,"p":15,"s":16,"cl":17,"ar":18,"k":19,"ca":20,"sc":21,"ti":22,"v":23,"cr":24,"mn":25,"fe":26,"co":27,"ni":28,"cu":29,"zn":30,"ga":31,"ge":32}

axes = None

def natom(model,lines,lam_min,lam_max):
    mdl = pd.HDFStore(model)
    lines = pd.HDFStore(lines)
    lambdas = np.asarray(u.Quantity(mdl.get_node("runner").last_interaction_in_nu.values.read(), 'Hz').to('angstrom', u.spectral()))
    lmds = (lambdas > lam_min)*(lambdas < lam_max)
    lines_id = mdl.get_node("runner").last_line_interaction_out_id.values.read()
    ids = lines_id[lmds]
    Z = lines.lines.ix[ids].values[:,1].astype(np.int)

    return Z

def lineshist(model, lines, lam_min, lam_max,fig=None):

    if fig is None:
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
    else:
        axes = fig.get_axes()
        ax1 = axes[0]

        Z=natom(model,lines, lam_min, lam_max)
        bins=np.linspace(0.5,32.5,33)
        ax1.hist(Z, bins = bins)
        ax1.set_xlabel("Atomic Number")
        ax1.legend()
    #fig = plt.gcf()
    #return fig

def create_color_list(start, end, cmap_name):
    jet = cm = plt.get_cmap(cmap_name)
    cNorm  = colors.Normalize(vmin=start, vmax=end)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    _tmp = []
    for i in np.arange(start, end, 1):
        _tmp.append(scalarMap.to_rgba(i))
    return _tmp

def lineskromer(model, lines,lam_min,lam_max,fig,nbins=300):

    if fig is None:
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
    else:
        axes = fig.get_axes()
        ax2 = axes[1]

        x=np.linspace(lam_min,lam_max,nbins)
        xright=x[1:]
        xleft=x[:-1]
        y=np.zeros((32,nbins-1))
        bins=np.linspace(0.5,32.5,33)
        for i in xrange (nbins-1):
            Z=natom(model, lines, xleft[i],xright[i])
            xs,ys=np.histogram(Z,bins)
            y[:,i]=xs

        _jet=cmx.jet
        cNorm  = colors.Normalize(vmin=1, vmax=32)
        map_array = np.linspace(1, 31, 32)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=_jet)
        scalarMap.set_array(map_array)
        xcenters=(xright+xleft)*0.5
        my_map = create_color_list(0,31,'jet')
        ax2.stackplot(xcenters,y,colors=my_map)
        cbar = fig.colorbar(scalarMap, ax = ax2, orientation='horizontal', boundaries = np.linspace(0, 31, 32))
        cbar.set_ticks(np.linspace(1, 32, 33))
        labels=["h","he","li","be","b","c","n","o","f","ne","na","mg","al","si","p","s","cl","ar","k","ca","sc","ti","v","cr","mn","fe","co","ni","cu","zn","ga","ge"]
        cbar.set_ticklabels(labels)
