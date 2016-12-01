#!/bin/sh
#$ -S /usr/bin/bash
# change to current working directory
#$ -cwd
# do not merge STDOUT and STDERR
#$ -j n
# paralellization
#$ -pe sm NTHREADS
# resource list
#$ -l arch=*-amd64
##$ -l h_vmem=4G
#$ -l qname=pascal
#$ -l h_cpu=48:00:00
# job name (arbitrary)
#$ -N tardis
# notifcation options
#$ -m beas
#$ -M talytha@mpa-garching.mpg.de
# stdout and stderr redirection
#$ -o /afs/mpa/home/talytha/logs/$JOB_NAME.$JOB_ID.out
#$ -e /afs/mpa/home/talytha/logs/$JOB_NAME.$JOB_ID.err

echo "Preparing module system..."
source /usr/common/appl/modules-tcl/init/bash
echo "load python"
module load python

echo "Starting tardis..."
export OMP_NUM_THREADS=${NSLOTS}
echo "Allocated slots:  " ${NSLOTS}
echo "OMP_NUM_THREADS: " ${OMP_NUM_THREADS}
executablefile filename rootname epoch
echo "done"
