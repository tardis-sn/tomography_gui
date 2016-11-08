#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
import pandas as pd
abunds_idents={"h":1,"he":2,"li":3,"be":4,"b":5,"c":6,"n":7,"o":8,"f":9,"ne":10,"na":11,"mg":12,"al":13,"si":14,"p":15,"s":16,"cl":17,"ar":18,"k":19,"ca":20,"sc":21,"ti":22,"v":23,"cr":24,"mn":25,"fe":26,"co":27,"ni":28,"cu":29,"zn":30,"ga":31,"ge":32}

axes = None

def natom(model,lines,lam_min,lam_max):
    #get Z from mdl:
    #lambdas=np.array(mdl.last_line_interaction_angstrom.base)
    #lmds=(lambdas>lam_min)*(lambdas<lam_max)
    #indices=np.argwhere(lmds==True).reshape(-1)
    #ids=[mdl.last_line_interaction_out_id[i] for i in indices]
    #Z=np.array([int(mdl.atom_data.lines.ix[id].values[1]) for id in ids])

    #get Z from hdf file
    mdl=pd.HDFStore(model)
    lines_=pd.HDFStore(lines)

    lambdas=mdl.get_node("last_line_interaction_angstrom").values.read()
    lmds=(lambdas>lam_min)*(lambdas<lam_max)
    #indices=np.argwhere(lmds==True).reshape(-1)
    lines_id=mdl.get_node("last_line_interaction_out_id").values.read()
    #ids=[lines_id[i] for i in indices]
    ids=lines_id[lmds]
    #Z = np.array([int(lines_.lines.ix[id].values[1]) for id in ids])
    Z = lines_.lines.ix[ids].values[:,1].astype(np.int)
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

    #Z=natom(mdl,lam_min,lam_max)
        x=np.linspace(lam_min,lam_max,nbins)
        xright=x[1:]
        xleft=x[:-1]
        y=np.zeros((32,nbins-1))
        bins=np.linspace(0.5,32.5,33)
        for i in xrange (nbins-1):
            Z=natom(model, lines, xleft[i],xright[i])
            #bins=(xleft[i],xright[i])
            xs,ys=np.histogram(Z,bins)
            y[:,i]=xs

        _jet=cmx.jet
        cNorm  = colors.Normalize(vmin=1, vmax=32)
        map_array = np.linspace(1, 31, 32)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=_jet)
        scalarMap.set_array(map_array)
        xcenters=(xright+xleft)*0.5
        #fig,ax=plt.subplots()
        my_map = create_color_list(0,31,'jet')
        ax2.stackplot(xcenters,y,colors=my_map)
        cbar = fig.colorbar(scalarMap, ax = ax2, orientation='horizontal', boundaries = np.linspace(0, 31, 32))
        #labels=[]
        cbar.set_ticks(np.linspace(1, 32, 33))
        labels=["h","he","li","be","b","c","n","o","f","ne","na","mg","al","si","p","s","cl","ar","k","ca","sc","ti","v","cr","mn","fe","co","ni","cu","zn","ga","ge"]
#        for k,i in abunds_idents.items():
#            labels.append(k)
        cbar.set_ticklabels(labels)
    #return fig
