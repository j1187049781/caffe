caffe_root = '/home/ubuntu/lab/caffe/'
sovler_protxt_file = caffe_root + '/models/my_finetune_flickr_style/sovler.prototxt'
train_prototxt = caffe_root + '/models/my_finetune_flickr_style/train.prototxt'
val_prototxt = caffe_root + '/models/my_finetune_flickr_style/val.prototxt'
weight_dir = caffe_root + 'models/my_finetune_flickr_style/'
imageNet_weights = caffe_root + \
                   'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
NUM_STYLE_LABELS=5
finetuned_weights='/home/ubuntu/lab/caffe/models/my_finetune_flickr_style/weights.style-net.caffemodel'
deploy_style_recognition_net_filename=caffe_root + '/models/my_finetune_flickr_style/deploy.prototxt'
flickr_style_mean_npy_path = '/home/ubuntu/lab/caffe/flickr_style/mean.npy'
imagenet_mean_bin_path = \
    '/home/ubuntu/lab/caffe/data/ilsvrc12/imagenet_mean.binaryproto'