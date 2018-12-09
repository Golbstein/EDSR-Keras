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

        
def sr_gen(image_datagen, path, mode = 'train', bs = 8, img_size = 256, add_text = False, scale = 4):
    
    if mode == 'train':
        label_generator = image_datagen.flow_from_directory(path+'JPEGImages/', target_size=(img_size, img_size), batch_size=bs,
                                                            class_mode = None, classes = ['train'], subset = 'training',
                                                            seed = 7)
        mask_generator = image_datagen.flow_from_directory(path, target_size=(img_size, img_size), batch_size=bs,
                                                                  class_mode = None, classes = ['SegmentationClassAug'], 
                                                                  subset = 'training', seed = 7, color_mode = 'grayscale')
    else:
        label_generator = image_datagen.flow_from_directory(path+'JPEGImages/', target_size=(img_size, img_size), batch_size=bs,
                                                            class_mode = None, classes = ['train'], subset = 'validation',
                                                            seed = 7)
        mask_generator = image_datagen.flow_from_directory(path, target_size=(img_size, img_size), batch_size=bs,
                                                              class_mode = None, classes = ['SegmentationClassAug'],
                                                              subset = 'validation', seed = 7, color_mode = 'grayscale')

        
    # combine generators into one which yields image and masks
    return perceptual_generator(zip(label_generator, mask_generator), add_text, img_size, scale), label_generator.n//bs

    
def perceptual_generator(image_datagen, add_text = False, img_size = 512, scale = 4):
    while True:
        y, mask = next(image_datagen)
        x = np.zeros((y.shape[0],y.shape[1]//scale,y.shape[2]//scale,3))
        sample_weights = np.zeros(y.shape)
        if add_text:
            for i in range(len(x)):
                text_image = generate_text_on_image(img_size)
                y[i] = cv2.add(y[i].astype('uint8'), text_image)
                x[i] = cv2.resize(y[i], (img_size//scale, img_size//scale))
                #sample_weights[i] = 1+(text_image!=0).astype('float32')*9
        else:
            for i in range(len(x)):
                x[i] = cv2.resize(y[i], (img_size//scale, img_size//scale))
                #sample_weights[i] = 1
        mask = np.reshape(mask, (-1, img_size**2, 1)).astype('int32')
        mask[mask>20]=21
        org_shape = mask.shape
        res = np.array([label2weight(yy) for yy in mask.flatten()])
        sample_weights = np.reshape(np.array(res), org_shape[:2]).astype('float32')
        onehot_mask = to_categorical(mask, num_classes=22)[:,:,:-1] # without last label 
        targ = np.zeros((x.shape[0], 1), dtype = 'float32') # loss should strive to zero
        #yield [x, y, sample_weights], [targ, targ, targ, targ, targ, targ]
        yield [x, y, onehot_mask, sample_weights], [targ, targ, targ, targ, targ, targ]