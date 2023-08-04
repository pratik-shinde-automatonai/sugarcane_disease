import cv2
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
#from train import HEIGHT,WIDTH

H = 512
W = 512

COLORMAP = [[53,17,236],[238,32,69],[250,250,55],[252,39,180]]
CLASSES = ['leaf_scald','red_strip','rust','white_mot']

def read_paths(PATH):
    #Reading images paths    
    for (_,_,files) in os.walk(PATH):
        image_list= files
    image_list.sort()
    for index,element in enumerate(image_list):
        image_list[index] = os.path.join(PATH,element)
    
    return image_list


#This function will create labels for us
def convert_to_segmentation_mask(mask):
    # Pascal VOC uses an RGB image to encode the segmentation mask for that image. RGB values of a pixel
    # encode the pixel's class.
    #
    # AutoAlbument requires a segmentation mask to be a NumPy array with the shape [height, width, num_classes].
    # Each channel in this mask should encode values for a single class. Pixel in a mask channel should have
    # a value of 1.0 if the pixel of the image belongs to this class and 0.0 otherwise.
    height, width = mask.shape[:2]
    segmentation_mask = np.zeros((height, width, len(COLORMAP)), dtype=np.float32)
    for label_index, label in enumerate(COLORMAP):
        segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(float)
    return segmentation_mask

def load_data(image_list,mask_list):
    '''Loads the data and creates training validation and test lists of paths.
    '''
    train_x, valid_x = train_test_split(image_list, test_size=0.20, random_state=42)
    train_y, valid_y = train_test_split(mask_list, test_size=0.20, random_state=42)
       
    return (train_x, train_y), (valid_x, valid_y)

def read_image(x):
    '''Function reads the image resizes , normalizes and converts into float32.'''
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (W, H))
    x = x.astype(np.float32)
    return x

def read_mask(x):
    '''Function reads the mask'''
    x = cv2.imread(x)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (W, H))
    x = convert_to_segmentation_mask(x)
    x = x.astype(np.int32)
    return x

def tf_dataset(x, y, batch=8):
    '''Function creates a tfds dataset using method from_tensor_slices,
    preprocesses the dataset by shuffling mapping the preprocess function and creating batches
    '''
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.map(preprocess,num_parallel_calls = tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    it = dataset.__iter__()
    return it

def preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()

        image = read_image(x)
        mask = read_mask(y)

        return image, mask

    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.int32])
    return image, mask


def postprocess_batchsize_one(output):
    output = output[0]
    output = np.where(output >0.5, 255, 0)
    return output

def create_rgb_mask(mask, color_map):
    '''
    Input mask nparray of 0 and 1 
    Color map python list of RGB values [[255,0,0],[255,255,0],[0,255,255]]
    Number of channels of mask and number of RGB values must match
    '''
    overlay_image = np.zeros_like(mask)
    
    _,_,channels = mask.shape
    
    for i in range(channels):
        positions  = np.argwhere(mask[:,:,i] > 0)
        
        for pos in positions:
            overlay_image[pos[0],pos[1],:] = color_map[i]
    
    return overlay_image

def overlay_result(input_image , mask, color_map, alpha):
    '''
    input_image: RGB (numpy image)
    mask: RGB mask (numpy image)
    color_map: python list eg: COLORMAP = [[255,0,0],[255,255,0],[0,255,255]]
    alpha: TO set the opacity
    '''
    
    color_map=np.array(color_map)
    beta = 1.0 - alpha
    
    mask = cv2.resize(mask,(input_image.shape[1],input_image.shape[0]),interpolation=cv2.INTER_BITS)
    
    _,_,channels = mask.shape
    
    for i in range(channels):
        positions  = np.argwhere(mask[:,:,i] > 0)
        
        for pos in positions:
            input_image[pos[0],pos[1],:] =  np.uint8(alpha*(color_map[i])+beta*(input_image[pos[0],pos[1],:]))
            
    return input_image
