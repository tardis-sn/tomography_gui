#!/afs/mpa/home/talytha/python-virtualenvs/tardis-current/bin/python
import tardis
from tardis import run_tardis
import argparse

def run(filename, runid, nepoch):
    mdl = run_tardis(filename)
    complete_file = open('completed_run_%05d_%d.txt' % (runid, nepoch), 'w+')
    complete_file.close()
    mdl.save_spectra("spec_%05d_%d.dat" % (runid, nepoch))
    #mdl.to_hdf5("model_%05d_%d.h5" % (runid, nepoch))
    #mdl.atom_data.lines.to_hdf("lines_%05d_%d.h5" % (runid, nepoch), "lines")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Tardis')
    parser.add_argument('filename', type=str, help='Tardis filename')
    parser.add_argument('rootname', type=int, help='rootname of the Tardis file')
    parser.add_argument('epoch', type=int, help='epoch of the Tardis file')
    args = parser.parse_args()
    run(args.filename, args.rootname, args.epoch)
