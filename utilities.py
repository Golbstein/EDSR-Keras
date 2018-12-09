import cv2
import random
import keras
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt


def random_crop(image, crop_shape):
    if (crop_shape[0] < image.shape[1]) and (crop_shape[1] < image.shape[0]):
        x = random.randrange(image.shape[1]-crop_shape[0])
        y = random.randrange(image.shape[0]-crop_shape[1])
        
        return image[y:y+crop_shape[1], x:x+crop_shape[0], :]
    else:
        image = cv2.resize(image, crop_shape)
        return image

def test_edsr(model, x):
    p = model.predict(x[None])
    p = np.clip(p, 0, 255)
    plt.figure(figsize=(10,10))
    plt.subplot(121)
    plt.imshow(x.astype('uint8'))
    plt.title('Low Res')
    plt.subplot(122)
    plt.imshow(p[0].astype('uint8'))
    plt.title('Super Res')


def ssim(y_true, y_pred):
    return K.expand_dims(tf.image.ssim(y_true, y_pred, 255.), 0)

content_w=[0.1, 0.8, 0.1]
def content_fn(x):
    content_loss = 0
    n=len(content_w)
    for i in range(n): content_loss += RMSE(x[i]-x[i+n]) * content_w[i]
    return content_loss

def RMSE(diff): 
    return K.expand_dims(K.sqrt(K.mean(K.square(diff), [1,2,3])), 0)
