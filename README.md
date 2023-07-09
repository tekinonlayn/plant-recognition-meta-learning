# plant-recognition-meta-learning
For a journal article.

# Code Repository

This repository contains the implementation of various components and algorithms related to the First-Order Model Agnostic Meta-Learning (FOMAML) algorithm. It includes the following files:

## MAML.py

File: [MAML.py](MAML.py)

Description: Implementation of the First-Order Model Agnostic Meta-Learning (FOMAML) algorithm. This file contains the main script that implements the FOMAML algorithm using ResNet18 and CBAM attention module. It includes the training and evaluation loops for meta-learning tasks.

## ResNet_CBAM.py

File: [ResNet_CBAM.py](ResNet_CBAM.py)

Description: Implementation of the ResNet18 model using the Convolutional Block Attention Module (CBAM). This file provides the ResNet18_CBAM class, which incorporates the CBAM attention module into the ResNet18 architecture. The code for this module is inspired by [elbuco1/CBAM](https://github.com/elbuco1/CBAM).

## ResNet18.py

File: [ResNet18.py](ResNet18.py)

Description: Implementation of the ResNet18 model. This file provides the ResNet18 class, which is an implementation of the ResNet18 architecture for image classification. The code for this implementation is inspired by [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar).

## feature-map-tSNE.py

File: [feature-map-tSNE.py](feature-map-tSNE.py)

Description: Visualization of the feature maps using the t-SNE algorithm. This file demonstrates how to extract feature maps from a model (such as ResNet18_CBAM) and visualize them using t-SNE for dimensionality reduction and scatter plot visualization.

## heatmap-visualization.py

File: [heatmap-visualization.py](heatmap-visualization.py)

Description: Heatmap visualization for the attention features of the ResNet18_CBAM model. This file provides an example of how to extract attention features from the ResNet18_CBAM model and visualize them as heatmaps, providing insights into the model's attention mechanisms.

Please refer to the individual files for more detailed information, usage instructions, and specific code implementations.

## License

This code repository is licensed under the [MIT License](LICENSE).

Feel free to customize this README file based on your specific repository structure and requirements. You can include additional sections such as installation instructions, usage examples, and acknowledgments as needed.
