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
COPY data/config/* /data/config/
COPY data/model_character/* /data/model_character/
Copy data/model_pretrained/* /data/model_pretrained/
COPY data/model_entity/* /data/model_entity/

ADD gru.py /
ADD gru_pretrained.py /
ADD gru_entitiylibrary.py /
ADD torch_preprocess.py /
ADD read_files.py /
ADD rnn_character.py
ADD rnn_characters_pretrained.py /
ADD rnn_character_entitylibrary.py /
ADD main.py /


# to install dependencies with pip, see the following example
Run pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
Run pip3 install torchvision
RUN pip3 install nltk
RUN pip3 install scikit-learn
RUN pip3 install numpy
RUN pip3 install gensim
RUN pip3 install flair

# This is the command that will be run when you start the container
# Here it's a python script, but it could be a shell script, etc.
CMD [ "python", "./main.py" ]
