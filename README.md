# Contents

1. [Usage](#1-usage)
2. [Cite](#2-cite)

# 1. Usage

## Requirements

Python version 3.10+.
See [requirements.txt](./requirements.txt) for full list of packages.

## Installation instructions
0) This repo should be your working directory.
1) Create new environment with python>=3.10 and activate it. 
2) Install packages.

Example:
```
conda create --name safari_env python=3.12
conda activate safari_env
pip install -e .
```
## Current limitations

The current version is implemented in numpy and runs on CPU only.  Updates for GPU coming soon.

## Get started

For a walkthrough of custom SSM instantiation and the options in this package, see [scripts/Example_Usage.ipynb](scripts/Example_Usage.ipynb).

## Contact us

This codebase is written and maintained by @echbaba and @MelJWhite. 

# 2. Cite

If you use this code, please cite the following paper:

```
@article{
  babaei2025safari,
  title={Sa{FAR}i: State-Space Models for Frame-Agnostic Representation},
  author={Hossein Babaei and Mel White and Sina Alemohammad and Richard Baraniuk},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2025},
}
```
