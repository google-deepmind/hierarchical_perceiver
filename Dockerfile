FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04

ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 32
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# install system dependencies
RUN apt-get clean
RUN apt-get update --fix-missing
RUN apt-get install -y \
            python3.7 \
            python3.7-dev \
            python3.7-distutils \
            python3-venv \
            python3-pip \
            wget \
            bzip2 \
            ca-certificates \
            git \
            unzip \
            apt-transport-https \
            ca-certificates \
            gnupg \
            curl \
            gcc \
            mono-mcs \
            build-essential \
            vim \
    && apt-get clean

# Little hack to get around problems with outdated NVIDIA signing keys
# https://github.com/NVIDIA/nvidia-docker/issues/1632
RUN rm /etc/apt/sources.list.d/cuda.list

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
RUN apt-get update --fix-missing
RUN apt-get install -y \
            google-cloud-cli \
    && apt-get clean

# Create a non-root user
ARG username=github_user
ARG uid=1000
ARG gid=100
ENV USER $username
ENV UID $uid
ENV GID $gid
ENV HOME /home/$USER
RUN useradd github_user -u $UID -g $GID --create-home \
    && rm -f $HOME/.bashrc
USER $USER

ENV CUDA_VISIBLE_DEVICES 0,1
ENV CUDA_HOME /usr/local/cuda
ENV FORCE_CUDA 1

# create conda environment, install some conda libraries
SHELL ["/bin/bash", "-l", "-c"]

WORKDIR $HOME

# Copy in the code to open-source
COPY --chown=$uid:$gid ./requirements_all_with_hashes.txt $HOME/
COPY --chown=$uid:$gid ./requirements_gpu_with_hashes.txt $HOME/
COPY --chown=$uid:$gid ./requirements_cpu_with_hashes.txt $HOME/
COPY --chown=$uid:$gid ./perceiver_helpers.py $HOME/
COPY --chown=$uid:$gid ./perceiver_blocks.py $HOME/
COPY --chown=$uid:$gid ./perceiver_test.py $HOME/
COPY --chown=$uid:$gid ./perceiver.py $HOME/

# install any extra Python requirements defined in requirements.txt
RUN pip3 install --require-hashes -r requirements_all_with_hashes.txt
RUN pip3 install --require-hashes -r requirements_gpu_with_hashes.txt
