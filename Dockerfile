
FROM ubuntu:20.04

RUN conda init bash 
RUN conda init fish

RUN conda install -c conda-forge mamba -y

RUN mamba install -c conda-forge fish jupyterlab git git-lfs git-crypt -y

RUN echo y | pip install torch --index-url https://download.pytorch.org/whl/cu118

RUN git lfs install 
RUN git config --global credential.helper store

RUN mkdir \app

ADD requirements.txt \app
ADD test_N_Body.yml \app
