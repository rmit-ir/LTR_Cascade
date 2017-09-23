# LTR_Cascade #

This repository is home to a reference implementation of the cascade ranking
model in the SIGIR '17 paper "Efficient Cost-Aware Cascade Ranking for
Multi-Stage Retrieval".

If you use this package in your work, please cite the following paper:

```
@inproceedings{chen_efficient_2017,
 author = {Chen, Ruey-Cheng and Gallagher, Luke and Blanco, Roi and Culpepper, J. Shane},
 title = {Efficient Cost-Aware Cascade Ranking in Multi-Stage Retrieval},
 booktitle = {Proceedings of {SIGIR} '17},
 year = {2017},
 pages = {445--454},
 publisher = {ACM}
} 
```


## Get Started ##

To compile our feature processing binaries:

    ./init-script/build.sh

This will also install/compile all the external dependencies such as Indri and
WANDbl.  Programs and header files will be installed at `external/local/bin`.

To install python dependencies:

    pip install -r python/requirements.txt

Other bash scripts and Makefiles under `experiments` should work out of the box.


## Ranking Experiments ##

* [Yahoo! Set 1](experiments/Yahoo_Set1/)
* [Gov2](experiments/Gov2/)
