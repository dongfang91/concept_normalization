# use a container for python 3
FROM python:3.6

# make a directory for data resources
RUN mkdir /data/
RUN mkdir /data/AskAPatient/
RUN mkdir /data/TwADR-L/
RUN mkdir /data/model/


# this will copy the the required files to the Docker container
COPY data/AskAPatient/* /data/AskAPatient/
COPY data/TwADR-L/* /data/TwADR-L/
COPY data/model/* /data/model/
ADD gru.py /
ADD rnn_baseline.py /
ADD term_matching_baseline.py /
ADD torch_preprocess.py /

# to install dependencies with pip, see the following example
Run pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
Run pip3 install torchvision
RUN pip3 install nltk
RUN pip3 install scikit-learn
RUN pip3 install numpy
RUN pip3 install gensim

# This is the command that will be run when you start the container
# Here it's a python script, but it could be a shell script, etc.
CMD [ "python", "./rnn_baseline.py" ]
