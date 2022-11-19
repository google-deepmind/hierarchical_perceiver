# Hierarchical Perceiver (HiP): Code Release

This repository contains code for the released paper
["HiP: Hierarchical Perceiver"](https://arxiv.org/abs/2202.10890) by Joao
Carreira, Skanda Koppula, Daniel Zoran, Adria Recasens, Catalin Ionescu, Olivier
Henaff, Evan Shelhamer, Relja Arandjelovic, Matt Botvinick, Oriol Vinyals, Karen
Simonyan, Andrew Zisserman, Andrew Jaegle.

This repository includes the official Jax implementation of the HiP
architecture, and a slightly improved implementation of the
[PerceiverIO architecture](https://github.com/deepmind/deepmind-research/tree/master/perceiver).

General perception systems such as Perceivers can process arbitrary modalities in any combination and are able to handle up to a few hundred thousand inputs. They achieve this generality by using exclusively global attention operations. This however hinders them from scaling up to the inputs sizes required to process raw high-resolution images or video. In this paper, we show that some degree of locality can be introduced back into these models, greatly improving their efficiency while preserving their generality. To scale them further, we introduce a self-supervised approach that enables learning dense low-dimensional positional embeddings for very large signals. We call the resulting model a Hierarchical Perceiver (HiP), released in this repository.

## Installation and Sample Usage

To create and activate a virtualenv and install all necessary dependencies, run:

```
python3 -m venv /tmp/hip_venv
source /tmp/hip_venv/bin/activate
pip3 install pip setuptools wheel
pip3 install -r requirements_gpu.txt
```

`requirements_gpu.txt` will require a working CUDA installation, and `nvidia-cuda-toolkit` installed. Otherwise, you can replace this with `requirements_cpu.txt` which does not have this dependency.

Running `python3 -m unittest -v perceiver_test` after this will run a suite of
tests that demonstrates running the HiP architecture on sample multimodal data.

We also include a Dockerfile that builds and installs all necessary packages in
a standardized image. To build the Docker container, run:

```
docker build . -t hip_opensource:latest
```

And to jump into the Docker container:

```
# Adjust based on your device availability
DEVICE_FLAGS="--gpus all  --device /dev/nvidia0 --device /dev/nvidia1  --device /dev/nvidia-uvm  --device /dev/nvidia-uvm-tools --device /dev/nvidiactl"

# For GPU
docker run -u root --shm-size 32G -it --rm --entrypoint /bin/bash  ${DEVICE_FLAGS} hip_opensource:latest

# For CPU
docker run -u root --shm-size 32G -it --rm --entrypoint /bin/bash  hip_opensource:latest
```

From here, you can run `python3 -m unittest -v perceiver_test` like before. On
our 8-core Intel Xeon Linux machine with a Quadro P1000 GPU, this takes 143
seconds to finish running.

The core HiP architecture implementation can be found in `perceiver.py`.

## Pre-trained checkpoints and Colab

Around January 2023, we will release pre-trained checkpoints in [this](https://storage.googleapis.com/dm-detcon/dm-hierarchical-perceiver) Google Cloud Storage bucket, and
a Colab demonstrating inference using these checkpoints.

## Citing this work

If you use this code in your work, please consider referencing our work:

```
@article{carreira2022hierarchical,
  title={Hierarchical perceiver},
  author={Carreira, Joao and Koppula, Skanda and Zoran, Daniel and Recasens, Adria and Ionescu, Catalin and Henaff, Olivier and Shelhamer, Evan and Arandjelovic, Relja and Botvinick, Matt and Vinyals, Oriol and others},
  journal={arXiv preprint arXiv:2202.10890},
  year={2022}
}
```

## License and disclaimer

Copyright 2022 DeepMind Technologies Limited

All software and materials are licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
