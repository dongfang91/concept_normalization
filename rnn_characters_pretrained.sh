#!/bin/bash
# Your job will use 1 node, 28 cores, and 168gb of memory total.
#PBS -q standard
#PBS -l select=1:ncpus=28:mem=168gb:pcmem=8gb:ngpus=1
### Specify a name for the job
#PBS -N rnn_character
### Specify the group name
#PBS -W group_list=nlp
### Used if job requires partial node only
#PBS -l place=pack:exclhost
### CPUtime required in hhh:mm:ss.
### Leading 0's can be omitted e.g 48:0:0 sets 48 hours
#PBS -l cput=672:00:00
### Walltime is how long your job will run
#PBS -l walltime=24:00:00
#PBS -e /home/u25/dongfangxu9/concept_normalization/log/rnn_pretrainederr
#PBS -o /home/u25/dongfangxu9/concept_normalization/log/rnn_pretrainedout

#####module load cuda80/neuralnet/6/6.0
#####module load cuda80/toolkit/8.0.61
module load singularity/2/2.6.0

cd $PBS_O_WORKDIR

singularity run --nv /extra/dongfangxu9/img/flair.img rnn_characters_pretrained.py

