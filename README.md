# Attention-Based Meta-Learning for Few-Shot Classification of Plant Diseases
## Abstract
Plant disease recognition using images is crucial in agriculture, but collecting such images is difficult and expensive. Classifying with few samples is thus necessary, but the performance of deep learning models used for classification typically declines with few samples. To address this issue, meta-learning approaches have been developed. Meta-learning approaches use episodic training methodology, in contrast to deep learning. Typically, convolutional networks are employed in meta-learning models for the purpose of feature extraction from samples. In this study, a Convolutional Block Attention Module (CBAM) attention mechanism was integrated into the ResNet18 model to facilitate feature extraction from plant images. The CBAM mechanism was used to give more weight to the critical and relevant parts of the image, similar to the way humans perceive images. Our proposed model utilized the Model-Agnostic Meta-Learning (MAML) algorithm, an optimization-based meta-learning method, to classify embeddings obtained through our approach with a classical linear classifier. Our model achieved exceptional performance, with an accuracy of 96.01% in 10-way 20-shot classification on the Plant Village dataset, outperforming previous state-of-the-art models. These findings suggest that our proposed model has potential for detecting plant diseases in agricultural settings.

## Code Repository
This repository contains the implementation of various components and algorithms related to the First-Order Model Agnostic Meta-Learning (FOMAML) algorithm. It includes the following files:

File: [MAML.py](MAML.py)

Implementation of the First-Order Model Agnostic Meta-Learning (FOMAML) algorithm. This file contains the main script that implements the FOMAML algorithm using ResNet18 and CBAM attention module. It includes the training and evaluation loops for meta-learning tasks.

File: [ResNet_CBAM.py](ResNet_CBAM.py)

Implementation of the ResNet18 model using the Convolutional Block Attention Module (CBAM). This file provides the ResNet18_CBAM class, which incorporates the CBAM attention module into the ResNet18 architecture. The code for this module is inspired by [elbuco1/CBAM](https://github.com/elbuco1/CBAM).

File: [ResNet18.py](ResNet18.py)

Implementation of the ResNet18 model. This file provides the ResNet18 class, which is an implementation of the ResNet18 architecture for image classification. The code for this implementation is inspired by [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar).

File: [feature-map-tSNE.py](feature-map-tSNE.py)

Visualization of the feature maps using the t-SNE algorithm. This file demonstrates how to extract feature maps from a model (such as ResNet18_CBAM) and visualize them using t-SNE for dimensionality reduction and scatter plot visualization.

File: [heatmap-visualization.py](heatmap-visualization.py)

Heatmap visualization for the attention features of the ResNet18_CBAM model. This file provides an example of how to extract attention features from the ResNet18_CBAM model and visualize them as heatmaps, providing insights into the model's attention mechanisms.

Please refer to the individual files for more detailed information, usage instructions, and specific code implementations.

## License

This code repository is licensed under the [MIT License](LICENSE).
