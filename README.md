# EDSR-Keras
EDSR Super-Resolution Implementation with Keras

Keras implementation of the paper **"Enhanced Deep Residual Networks for Single Image Super-Resolution" from CVPRW 2017, 2nd NTIRE**: [EDSR Paper](https://arxiv.org/abs/1707.02921)

![alt text](https://github.com/Golbstein/EDSR-Keras/blob/master/main.png)

# Model Architecture
![alt text](https://github.com/Golbstein/EDSR-Keras/blob/master/EDSR.png)

## Extensions
1. Training with multi loss - MAE + **VGG16 Perceptual Loss** 
2. float16 and float32 support
3. Keras Subpixel (Pixel-Shuffle layer) from: [Keras-Subpixel](https://github.com/tetrachrome/subpixel/blob/master/keras_subpixel.py)
4. ICNR weights initialization - [Checkerboard artifact free sub pixel convolution initialization](https://arxiv.org/pdf/1707.02937.pdf), credit also for @kostyaev for the implementation of the initializer here: https://github.com/kostyaev/ICNR


**Training from scratch in float16 with multi-loss doesn't work. Set to float32**

## Dependencies
* Python 3.6
* Keras>2.0.x
* keras-tqdm (pip install keras-tqdm)
