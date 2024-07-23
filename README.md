# HyNN-Synthetic-Images
## Abstract
In the field of machine learning, the absence of spatial structure in tabular data poses significant limitations
on the applicability of Convolutional Neural Networks (CNNs). To address this issue, various works have
emerged by converting such data into synthetic images, encapsulating feature similarities within a spatial
context and thus broadening the application of CNNs to this type of data. This study delves into various
techniques for converting tabular data into synthetic images to develop hybrid models that combine different
architectures. Specifically, we propose two hybrid neural networks: Hybrid Neural Network (HyNN), which
combines a CNN for analysing synthetic images and a Multi-Layer Perceptron (MLP) for tabular data; and
Hybrid Vision Transformer (HyViT), which employs a Vision Transformer (ViT) for analysing synthetic
images and a MLP for tabular data. Through experimentation, focused on a regression problem using a
MIMO indoor localization large-scale dataset, we benchmarked HyViT and various HyNN configurations
against classical regression algorithms. Notably, the HyViT model achieves the lowest Root Mean Squared
Error (RMSE) outperforming the HyNN counterparts, and the best classical regression model. These findings
underscore the potential of synthetic images and hybrid architectural innovations in enhancing deep neural
network models in tabular data.
## Usage
### TINTOLib
TINTOlib is a state-of-the-art library that wraps the most important techniques for the construction of Synthetic Images from [Sorted Data](https://www.jstatsoft.org/article/view/v059i10) (also known as Tabular Data).

### HyNN
![HyNN Architecture](https://github.com/DCY1117/HyNN-Synthetic-Images/blob/main/Images/HyNN_CNN_MLP.png)
### HyViT
![HyViT Architecture](https://github.com/DCY1117/HyNN-Synthetic-Images/blob/main/Images/HyNN_ViT%2BMLP.png)

### Dataset
Original dataset: https://ieee-dataport.org/open-access/ultra-dense-indoor-mamimo-csi-dataset

The dataset used is the DIS scenario with 8 antennas.

Synthetic images and results in the following folder:
https://upm365-my.sharepoint.com/:f:/g/personal/jiayun_liu_upm_es/Eqhp7Jj3L3pLnK75Jx66nDQB3zBMp319Nqa4cCjrZmSSxw?e=yBxH2H

### Training and testing Scripts
For classical models, execute:

```bash
python Lazyregressor.py
```

For hybrid models, run the training scripts inside the training scripts folder. For example:
```bash
python training_scripts/HyCNN/DIS_8training_CNN_IGTD.py
```
Note: The images are generated automatically when executing the training scripts.
