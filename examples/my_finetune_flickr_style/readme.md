# Caffe：使用已有模型进行fine-tune
这个例子在bvlc_reference_caffenet(使用ImageNet 数据集 )上使用Flickr 的图片再训练
## step 1: 数据集的处理
* 这个例子的数据的使用下列形式 4. 数据已经做了标注，分为训练集，测试集
1. 数据来自于数据库（如LevelDB和LMDB）
2. 数据来自于内存
3. 数据来自于HDF5
4. 数据来自于图片
5. 数据来自于Windons
* 均值文件 [例子介绍](http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb)  
使用优点： 加快收敛速度、提高精度
1. 均值文件的生成
可以使用 caffe_root/tools/compute_imageg_mean.cpp
2. 使用和转换
## step 2: CaffeNet网络
CaffeNet是Alex的一种细微变形模型  
关于Net的可视化，可以使用[draw_net.py](../../python/draw_net.py)   
以下3个prototxt文件是在Python代码中实现生成  
* train.prototxt  
[代码生成](./CaffeNet.py)
* val.prototxt  
[代码生成](./CaffeNet.py)
* solver.prototxt  
[代码生成](./Solver.py)  
关于learning rate文档中这样解释  
We will also decrease the overall learning rate base_lr in the solver prototxt, but boost the lr_mult on the newly introduced layer. The idea is to have the rest of the model change very slowly with new data, but let the new layer learn fast. Additionally, we set stepsize in the solver to a lower value than if we were training from scratch, since we’re virtually far along in training and therefore want the learning rate to go down faster. Note that we could also entirely prevent fine-tuning of all layers other than fc8_flickr by setting their lr_mult to 0. [ref](http://caffe.berkeleyvision.org/gathered/examples/finetune_flickr_style.html)
## step 3: train
为了方便可视化训练模型的正确率变化，这个例子使用方法2
* method 1  
使用train 命令
* method 2  
使用Python [代码](./train.py)
## step 4: predict
1. 使用deploy.prototxt  
[使用Python接口来预测](./predictImg.py)  
2. 流程：  
set up --> and set up input preprocessing -->predict

## 关于train net ,test net ,deploy net 的对比
* train net-->test net  
去除drop layer, 增加probs(tpye: Softmax) layer
* test net-->deploy net
改变Data layer 层（第一维为度数1）

