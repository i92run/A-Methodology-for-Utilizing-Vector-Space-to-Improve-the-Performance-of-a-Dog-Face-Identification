# A Methodology for Utilizing Vector Space to Improve the Performance of a Dog Face Identification Model

This repository is the official implementation of *A Methodology for Utilizing Vector Space to Improve the Performance of a Dog Face Identification Model*. We propose a triplet-loss-based embedding learning framework that eliminates L2-normalization, enabling discrimination within the entire embedding space rather than on a unit hypersphere, and achieves improved open-set dog face identification through a novel loss function and two-stage training scheme.

<img src="/results_fig.png" width="672" height="320">


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


## Dataset

This work uses the DogFaceNet dataset:

Dataset reference: https://github.com/GuillaumeMougeot/DogFaceNet


## Citation

```bib
@article{yoon2021methodology,
  title={A methodology for utilizing vector space to improve the performance of a dog face identification model},
  author={Yoon, Bohan and So, Hyeonji and Rhee, Jongtae},
  journal={Applied Sciences},
  volume={11},
  number={5},
  pages={2074},
  year={2021},
  publisher={MDPI}
}
```
