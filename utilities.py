import cv2
import random
import keras
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import Sequence
from random import shuffle
from tqdm import tqdm

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


def loadImage(img, path_size, scale):
    img_size = int(path_size * scale)
    I = cv2.imread(img)
    I = random_crop(I, (img_size, img_size))
    y = I.copy()
    # Use different downsampling methods
    if np.random.randint(2): # x_scale sampling
        I = I[::scale, ::scale]
    else: #bilinear resizing
        I = cv2.resize(I, (path_size, path_size))
    return I, y

#flips a batch of images, flipMode is an integer in range(8)
def flip(x, flipMode):
    if flipMode in [4,5,6,7]:
        x = np.swapaxes(x,1,2)
    if flipMode in [1,3,5,7]:
        x = np.flip(x,1)
    if flipMode in [2,3,6,7]:
        x = np.flip(x,2)
    return x


# Load data to RAM (for training model using .fit method)
def load_train_data(IMAGES, N = np.inf, img_size = 96, scale = 2):
    '''
    N = number of images to return, 
    if M is the number of images in the folder and N>M then:
        return all the images in the folder
    '''
    x = []
    y = []
    for k, img in tqdm(enumerate(IMAGES)):
        if k>N: break
        I = cv2.imread(img)
        I = random_crop(I, (img_size,img_size))
        y.append(I.copy())
        
        # Use different downsampling methods
        if np.random.randint(2): # x_scale sampling
            I = I[::scale, ::scale]
        else: #bilinear resizing
            I = cv2.resize(I, (img_size//scale, img_size//scale))
        x.append(I)
        
    x = np.stack(x)
    y = np.stack(y)
    return x, y # return HighRes and LowRes images


#works with channels last
class ImageLoader(Sequence):
    
    #class creator, use generationMode = 'predict' for returning only images without labels
        #when using 'predict', pass only a list of files, not files and classes
    def __init__(self, files, path_size = 48, scale = 2, batchSize = 16, multi_loss=False, generationMode = 'train'):
        
        self.files = files
        self.batchSize = batchSize
        self.generationMode = generationMode
        self.path_size = path_size
        self.scale = scale
        self.multi_loss = multi_loss
        
        assert generationMode in ['train', 'predict']
            

    #gets the number of batches this generator returns
    def __len__(self):
        l,rem = divmod(len(self.files), self.batchSize)
        return (l + (1 if rem > 0 else 0))
    
    #shuffles data on epoch end
    def on_epoch_end(self):
        if self.generationMode == 'train':
            shuffle(self.files)
        
    #gets a batch with index = i
    def __getitem__(self, i):
        
        #x are images   
        #y are labels
        
        images = self.files[i*self.batchSize:(i+1)*self.batchSize]
        
        x,y = zip(*[loadImage(f, self.path_size, self.scale) for f in images])
        
        x = np.stack(x, axis=0) # Low Res
        y = np.stack(y, axis=0) # High Res
        
        #cropping and flipping when training
        if self.generationMode == 'train':
                        
            flipMode = random.randint(0,7) #see flip functoin defined above
            x = flip(x, flipMode)
            y = flip(y, flipMode)
            
        if self.generationMode == 'predict':
            return x
        elif self.multi_loss:
            return [x, y], [y, np.zeros((len(x), 1))]
        else:
            return x, y
