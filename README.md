# HyNN-Synthetic-Images
## Abstract
In the field of machine learning, the absence of spatial structure in tabular data poses significant limitations on the applicability of Convolutional Neural Networks (CNNs). To address this issue, various works have emerged by converting such data into images, encapsulating feature similarities within a spatial context and thus broadening the application of CNNs to this type of data. This study delves into various techniques for converting tabular data into synthetic images to develop hybrid models that combine different architectures. Specifically, we propose two novel hybrid architectures: Hybrid Neural Network (HyNN), which combines a Convolutional Neural Network for analysing synthetic images and a Multilayer Perceptron for tabular data; and Hybrid Vision Transformer (HyViT), which employs a Vision Transformer for analysing synthetic images and a Multilayer Perceptron for tabular data. Through rigorous experimentation, focused on a regression problem using a MIMO indoor localization dataset, we benchmarked HyViT against classical regression algorithms and various HyNN configurations. Notably, the HyViT model achieves the lowest RMSE outperforming the HyNN counterparts, and the best classical regression model KNN. These findings underscore the potential of synthetic images and hybrid architectural innovations in enhancing deep learning models, offering a promising avenue for future research and application in diverse fields.

## Usage
### TINTOLib
TINTOlib is a state-of-the-art library that wraps the most important techniques for the construction of Synthetic Images from [Sorted Data](https://www.jstatsoft.org/article/view/v059i10) (also known as Tabular Data).

### HyNN
![HyNN Architecture](https://github.com/DCY1117/HyNN-Synthetic-Images/blob/main/Images/HyNN_CNN%2BMLP.png)
### HyViT
![HyViT Architecture](https://github.com/DCY1117/HyNN-Synthetic-Images/blob/main/Images/HyNN_ViT%2BMLP.png)
### Dataset
Synthetic images and results in the following folder:
https://upm365-my.sharepoint.com/:f:/g/personal/jiayun_liu_upm_es/Eqhp7Jj3L3pLnK75Jx66nDQB3zBMp319Nqa4cCjrZmSSxw?e=yBxH2H

### Training Scripts
