# Chest X-Ray Pneumonia Detection

This project implements a deep learning model to accurately classify chest X-ray images for the detection of pneumonia. The core of this solution is built using the PyTorch framework.

## ## Technical Approach
To address the challenge of working with a specialized medical imaging dataset, this project leverages the power of **transfer learning**.

I utilized the **VGG16 architecture**, pre-trained on the extensive ImageNet dataset, as a robust feature extractor. The convolutional base of the model was **frozen** to preserve its sophisticated understanding of low-level features like edges, textures, and patterns. A new, custom classifier head was then built and attached to this base. Only this new head was trained on the X-ray dataset, allowing the model to adapt its powerful, generalized knowledge specifically for the task of identifying pneumonia.

To enhance the model's ability to generalize and prevent overfitting, a **data augmentation** pipeline was implemented. This process artificially expands the training dataset by applying random transformations—such as rotations, flips, and zooms—to the images, ensuring the model learns the core pathological indicators of pneumonia rather than memorizing specific image orientations.

## ## Current Development
The repository currently contains the complete transfer learning implementation. I am now actively developing a second version of this model: a **Convolutional Neural Network (CNN) built entirely from scratch**. The purpose of this second implementation is to gain a foundational understanding of how convolutional layers, pooling, and activation functions work in concert to learn a hierarchy of features directly from raw pixel data, without reliance on pre-trained weights.
