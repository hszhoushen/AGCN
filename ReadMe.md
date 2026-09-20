# Attentional Graph Convolutional Network (AGCN)

Official implementation for **“Attentional Graph Convolutional Network for Structure-aware Audio-Visual Scene Classification.”**

AGCN learns structure-aware representations from audio and visual inputs by making salient, semantically meaningful regions explicit as graphs. The repository contains the model code and experiment scripts for scene recognition.

## Paper

- **Authors:** Liguang Zhou, Yuhongze Zhou, Xiaonan Qi, Junjie Hu, Tin Lun Lam, and Yangsheng Xu
- **Venue:** *IEEE Transactions on Instrumentation and Measurement*, vol. 72, 2023
- **DOI:** [10.1109/TIM.2023.3260282](https://doi.org/10.1109/TIM.2023.3260282)
- **arXiv:** [2301.00145](https://arxiv.org/abs/2301.00145)

## Overview

Audio-visual scene understanding must reason across audio events, visual objects, and their spatial and semantic context. AGCN is an end-to-end framework that combines features from sound spectrograms and images, aggregates multi-level backbone features with an attention fusion mechanism, and models salient and contextual regions using graph convolution.

The framework constructs four complementary graphs:

- **SAG:** Salient Acoustic Graph
- **CAG:** Contextual Acoustic Graph
- **SVG:** Salient Visual Graph
- **CVG:** Contextual Visual Graph

These graphs capture informative acoustic and visual regions and their relationships for scene recognition. The paper evaluates audio-only, visual-only, and audio-visual recognition, and visualizes the learned graphs to illustrate the regions emphasized by the model.

## Repository Contents

- `model.py`, `model/`, and related modules: model implementation
- `train_gnn_sr.py`, `train_gnn_sr_single_gpu.py`, and `train_gnn_sr_multi_gpus.py`: training entry points
- `test.py` and `test_gnn_sr.py`: evaluation entry points
- `*.sh`: experiment launch examples for the datasets and settings used in this repository
- `data/`, `dataset.py`, and `utils/`: data loading and supporting utilities

## Datasets and Experiment Scripts

The repository includes experiment scripts for several audio and visual scene-recognition settings, including:

- Audio: ESC-10, ESC-50, and UrbanSound8K
- Visual: MIT67, Places365-7, Places365-14, NYU, and SUN RGB-D

The scripts are starting points, not plug-and-play commands: they use machine-specific Conda environment names and GPU IDs. Before running one, configure the environment, dataset locations, GPU selection, and any dataset-specific options for your machine. Dataset access and preparation may be subject to each dataset's own terms.

For example, the Places365-14 training recipe is in [`Places365-14.sh`](Places365-14.sh), and the Places365-7 recipe is in [`Places365-7.sh`](Places365-7.sh). Check the selected script and training entry point for the required arguments before launching an experiment.

> **Reproducibility note:** The repository currently does not include a pinned dependency file or a complete dataset setup guide. Results may depend on the dataset split, preprocessing, pretrained weights, and configuration used.

## Citation

If you use this work, please cite:

```bibtex
@article{zhou2023attentional,
  title   = {Attentional Graph Convolutional Network for Structure-aware Audio-Visual Scene Classification},
  author  = {Zhou, Liguang and Zhou, Yuhongze and Qi, Xiaonan and Hu, Junjie and Lam, Tin Lun and Xu, Yangsheng},
  journal = {IEEE Transactions on Instrumentation and Measurement},
  volume  = {72},
  year    = {2023},
  doi     = {10.1109/TIM.2023.3260282}
}
```

## License

This repository is released under the [MIT License](LICENSE).
