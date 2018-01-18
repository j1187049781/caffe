# coding: utf-8
import caffe
import numpy as np
def bin_to_npy(bin_path,npy_path):
    blob = caffe.proto.caffe_pb2.BlobProto()  # 创建protobuf blob
    data = open(bin_path, 'rb').read()  # 读入mean.binaryproto文件内容
    blob.ParseFromString(data)  # 解析文件内容到blob

    array = np.array(
        caffe.io.blobproto_to_array(blob))  # 将blob中的均值转换成numpy格式，array的shape （mean_number，channel, hight, width）
    mean_npy = array[0]# 一个array中可以有多组均值存在，故需要通过下标选择其中一组均值
    np.save(npy_path, mean_npy)
