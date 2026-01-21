# OV-SCAN
## Semantically Consistent Alignment for Novel Object Discovery in Open-Vocabulary 3D Object Detection

![Python Version](https://img.shields.io/badge/Python-3776AB?&logo=python&logoColor=white-blue&label=3.10%20%7C%203.11)&ensp;
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)&ensp;
[![Paper: arXiv](https://img.shields.io/badge/Paper-arXiv-green.svg)](https://arxiv.org/abs/2503.06435)&ensp;

> 📣 **Our paper has been accepted to ICCV 2025!**

OV-SCAN is an Open-Vocabulary 3D framework that enforces Semantically Consistent Alignment for Novel object discovery. It employs two core strategies:
1. Discovering precise 3D annotations
2. Filtering out low-quality or corrupted alignment pairs (arising from 3D annotation, occlusion-induced, or resolution-induced noise)

\
<img src="./assets/media/ov-scan-method.jpg" alt="Overall framework for OV-SCAN"/> 
<p align="center"><em>Overall framework for OV-SCAN</em></p>

## 📚 Documentation

Comprehensive guides are available to help you get started:

1. **[Installation](docs/1_INSTALL.md)** - System requirements and dependencies
2. **[Environment Setup](docs/2_ENV_SETUP.md)** - Docker setup and environment configuration
3. **[Data Setup](docs/3_DATA_SETUP.md)** - Download and organize training data
4. **[Training & Evaluation](docs/4_TRAIN_EVAL.md)** - Train and evaluate OV-SCAN models

---

## 📊 Performance

### NuScenes Open-Vocabulary 3D Object Detection

OV-SCAN achieves state-of-the-art performance on the NuScenes dataset for open-vocabulary 3D object detection:

| Method | mAP | NDS | Car | Ped. | Truck | Motorcyc. | Bicyc. | T. Cone | Bus | Barrier | Con.V. | Trailer |
|--------|-----|-----|-----|------|-------|-----------|--------|---------|-----|---------|--------|---------|
| OV-SCAN | 31.1 | 32.8 | 61.6 | 60.1 | 30.3 | 39.8 | 31.0 | 39.6 | 22.0 | 18.8 | 6.8 | 0.6 |
| OV-SCAN-Fusion* | 33.8 | 34.4 | 62.0 | 57.6 | 34.2 | 44.7 | 40.2 | 44.6 | 24.4 | 18.9 | 10.8 | 0.8 |

\* OV-SCAN-Fusion uses multi-view image fusion for enhanced performance

Pre-trained model checkpoints are available for download:
- **[Download Pretrained Models](https://drive.google.com/file/d/1_HL7BiAGPdd4rwGyMBe3GvM0iuDVuFm8/view?usp=sharing)** - Includes backbone weights and trained OV-SCAN models

---

## Citation
If you use OV-SCAN in your research, please cite our paper:

```bibtex
@inproceedings{chow2025ovscan,
  title     = {OV-SCAN: Semantically Consistent Alignment for Novel Object Discovery in Open-Vocabulary 3D Object Detection},
  author    = {Adrian Chow and Evelien Riddell and Yimu Wang and Sean Sedwards and Krzysztof Czarnecki},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025},
}
```
