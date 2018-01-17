import sys
from matplotlib import cbook

from EnvConst import *
from CaffeNet import style_net
import os
sys.path.insert(0, caffe_root + 'python')
import caffe

import numpy as np
from EnvConst import *
import matplotlib.pyplot as plt

def deprocess_net_image(image):
    image = image.copy()              # don't modify destructively
    image = image[::-1]               # BGR -> RGB
    image = image.transpose(1, 2, 0)  # CHW -> HWC
    image += [123, 117, 104]          # (approximately) undo mean subtraction

    # clamp values in [0, 255]
    image[image < 0], image[image > 255] = 0, 255

    # round and cast from float32 to uint8
    image = np.round(image)
    image = np.require(image, dtype=np.uint8)

    return image

def disp_preds(net, image, labels, k=5, name='style-recognition-net'):
    input_blob = net.blobs['data']
    net.blobs['data'].data[0, ...] = image
    probs = net.forward(start='conv1')['probs'][0]
    top_k = (-probs).argsort()[:k]
    print 'top %d predicted %s labels =' % (k, name)
    print '\n'.join('\t(%d) %5.2f%% %s' % (i+1, 100*probs[p], labels[p])
                    for i, p in enumerate(top_k))
if __name__=='__main__':
    # Load style labels to style_labels
    style_label_file = caffe_root + 'examples/finetune_flickr_style/style_names.txt'
    style_labels = list(np.loadtxt(style_label_file, str, delimiter='\n'))
    if NUM_STYLE_LABELS > 0:
        style_labels = style_labels[:NUM_STYLE_LABELS]
    print '\nLoaded style labels:\n', ', '.join(style_labels)

    assert os.path.exists(finetuned_weights)
    test_net = caffe.Net(val_prototxt, finetuned_weights, caffe.TEST)

    # with cbook.get_sample_data('/home/ubuntu/lab/caffe/data/flickr_style/images/13286845864_7c28844874.jpg') as image_file:
    #     image = plt.imread(image_file)
    # plt.imshow(image)
    # plt.show()
    test_net.forward()
    batch_index = 40
    image = test_net.blobs['data'].data[batch_index]
    plt.imshow(deprocess_net_image(image))
    plt.show()
    print 'actual label =', style_labels[int(test_net.blobs['label'].data[batch_index])]
    disp_preds(test_net,image,style_labels)
