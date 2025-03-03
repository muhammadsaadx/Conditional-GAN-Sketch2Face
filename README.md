# Sketch2Face: Facial Image Generation from Sketches

## Overview

Sketch2Face is a deep learning model that can generate realistic human face images from hand-drawn or computer-generated sketches. This implementation uses a conditional GAN architecture based on the pix2pix model, which has shown excellent results for image-to-image translation tasks.

## Dataset Preparation

The model was trained on a dataset of paired facial photos and corresponding sketches. The dataset preparation script combines photos and sketches side by side to create training samples.

## Model Architecture

The model uses a conditional GAN architecture with:

### Generator (U-Net)
- Encoder-decoder architecture with skip connections
- 8 downsampling layers and 7 upsampling layers
- Uses batch normalization and dropout for regularization

### Discriminator (PatchGAN)
- Classifies patches of the image as real or fake
- 4 downsampling layers
- Uses batch normalization and leaky ReLU activations

## Training

To train the model:

```bash
python train.py
```

Training parameters:
- Learning rate: 2e-4
- Batch size: 1
- L1 lambda: 100
- Optimizer: Adam (beta1=0.5)
- Random jittering for data augmentation

Training checkpoints are saved every 5000 steps and can be found in the `training_checkpoints` directory.

## Inference

To generate faces from sketches:

```bash
python inference.py --input path/to/sketch.jpg --output path/to/output.jpg
```

## Results

The model progressively improves during training. Here are some results at various training stages:

- Initial results (0k steps): Blurry images with basic facial features
- Mid-training (2k steps): Improved details and color accuracy
- Final results (4k steps): Realistic facial features and textures

The final model achieves:
- Generator GAN loss: 3.85
- Generator L1 loss: 0.58
- Discriminator loss: 0.03

## Saving and Loading Models

The trained model weights can be saved in both HDF5 and TensorFlow checkpoint formats:

```python
# Save the model
generator.save_weights('./model_weights.h5')  # HDF5 format
generator.save_weights('./model_weights.ckpt')  # TensorFlow checkpoint format

# Load the model
generator = Generator()
generator.load_weights('./model_weights.h5')
```

## References

- [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) (Pix2Pix paper)
- [TensorFlow Pix2Pix Tutorial](https://www.tensorflow.org/tutorials/generative/pix2pix)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The dataset used for training is derived from [person-face-sketches](dataset_url_here)
- The implementation is based on the TensorFlow Pix2Pix tutorial
