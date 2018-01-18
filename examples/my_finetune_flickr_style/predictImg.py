import sys

from CaffeNet import *
import os

import meanUtil

sys.path.insert(0, caffe_root + 'python')
import caffe

import numpy as np
from EnvConst import *
import matplotlib.pyplot as plt


model_weights=finetuned_weights

if __name__=='__main__':
    caffe.set_mode_cpu()

    dummy_data = L.DummyData(shape=dict(dim=[1, 3, 227, 227]))
    if not os.path.exists(deploy_style_recognition_net_filename):
        caffenet(data=dummy_data, train=False,num_classes=NUM_STYLE_LABELS,
                    classifier_name='fc8_flickr',deploy=True)
    net = caffe.Net(deploy_style_recognition_net_filename,  # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)  # use test mode (e.g., don't perform dropout)

    if not os.path.exists(flickr_style_mean_npy_path):
        meanUtil.bin_to_npy(imagenet_mean_bin_path,flickr_style_mean_npy_path)
    # load the mean ImageNet image (as distributed with Caffe) for subtraction
    mu = np.load(flickr_style_mean_npy_path)
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
    print 'mean-subtracted values:', zip('BGR', mu)

    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)  # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

    image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
    transformed_image = transformer.preprocess('data', image)
    plt.imshow(image)
    plt.show()

    net.blobs['data'].data[...] = transformed_image

    ### perform classification
    output = net.forward()
    output_prob = output['probs'][0]  # the output probability vector for the first image in the batch
    print 'predicted class is:', output_prob.argmax()

    style_label_file = caffe_root + 'examples/finetune_flickr_style/style_names.txt'
    style_labels = list(np.loadtxt(style_label_file, str, delimiter='\n'))
    if NUM_STYLE_LABELS > 0:
        style_labels = style_labels[:NUM_STYLE_LABELS]

    print 'output label:', style_labels[output_prob.argmax()]


