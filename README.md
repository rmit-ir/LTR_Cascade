# LTR_Cascade #

This repository is home to a reference implementation of the cascade ranking
model in the SIGIR '17 paper "Efficient Cost-Aware Cascade Ranking for
Multi-Stage Retrieval".

If you use this package in your work, please cite the following paper:

> Ruey-Cheng Chen, Luke Gallagher, Roi Blanco, and J. Shane Culpepper. 
> Efficient Cost-Aware Cascade Ranking for Multi-Stage Retrieval. In
> Proceedings of SIGIR '17, to appear.


## Get Started ##

To compile our feature processing binaries:

    ./init-script/build.sh

This will also install/compile all the external dependencies such as Indri and
WANDbl.  Programs and header files will be installed at `external/local/bin`.

To install python dependencies:

    pip install -r python/requirements.txt

Other bash scripts and Makefiles under `experiments` should work out of the box.


## Ranking Experiments ##

* [Yahoo! Set 1](experiments/Yahoo_Set1/README.md)
* [Gov2](experiments/Gov2/README.md)
