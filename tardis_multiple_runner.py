#!/usr/bin/env python
import numpy as np
import argparse
import glob
import shlex
import subprocess

def main(rootname):

    naepochs = glob.glob("tardis_%05d*.yml" % rootname)
    nepochs = len(naepochs)
    epochs =[]

    for i in xrange (nepochs):
        epoch =(naepochs[i].rsplit("_")[-1].strip()).rsplit(".")[-2].strip()
        print epoch
        epochs.append(int(epoch))

    return epochs

def write_submit(Nthreads, rootname, mode = 'local'):

    epochs = main(rootname)

    if mode == 'batch':
        defaultfile = 'mpa-pascal-defaul.cmd'
        for i in xrange(len(epochs)):

            filename = 'tardis_%05d_%d.yml' %(rootname, epochs[i])

            with open(defaultfile, 'r') as file:
                 filedata = file.read()

            filedata = filedata.replace('NTHREADS', str(Nthreads))
            filedata = filedata.replace('executablefile', './run_tardis.py')
            filedata = filedata.replace('filename', str(filename))
            filedata = filedata.replace('rootname', str(rootname))
            filedata = filedata.replace('epoch', str(epochs[i]))

            with open('mpa-pascal_%05d_%d.cmd' % (rootname, epochs[i]), 'w') as file:
                file.write(filedata)

            p = subprocess.Popen(shlex.split("qsub mpa-pascal_%05d_%d.cmd" % (rootname, epochs[i])))

    if mode == 'local':

      for i in xrange(len(epochs)):
            defaultfile = 'submit_local_default.sh'
            filename = 'tardis_%05d_%d.yml' %(rootname, epochs[i])

            with open(defaultfile, 'r') as file:
                filedata = file.read()

            filedata = filedata.replace('executablefile', './run_tardis.py')
            filedata = filedata.replace('filename', str(filename))
            filedata = filedata.replace('rootname', str(rootname))
            filedata = filedata.replace('epoch', str(epochs[i]))

            with open('submit_local_%05d_%d.sh' % (rootname, epochs[i]), 'w') as file:
                file.write(filedata)

            subprocess.Popen(shlex.split("chmod +x submit_local_%05d_%d.sh" % (rootname, epochs[i])))

            with open("tardis_%05d_%d_err.log" %(rootname, epochs[i]), 'w+') as err:
                subprocess.Popen(shlex.split("./submit_local_%05d_%d.sh" % (rootname, epochs[i])), stdout = err , stderr = err )

#if __name__ == "__main__":
#    write_submit(Nthreads, rootname, mode)

write_submit(16,4)
