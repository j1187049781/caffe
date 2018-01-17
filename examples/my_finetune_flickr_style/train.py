# coding:utf-8
import sys

import os

from EnvConst import *
from CaffeNet import style_net

sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
import matplotlib.pyplot as plt
def run_solvers(niter, solvers, disp_interval=10):
    """Run solvers for niter iterations,
       returning the loss and accuracy recorded each iteration.
       `solvers` is a list of (name, solver) tuples."""
    blobs = ('loss', 'acc')
    loss, acc = ({name: np.zeros(niter) for name, _ in solvers}
                 for _ in blobs)
    for it in range(niter):
        for name, s in solvers:
            s.step(1)  # run a single SGD step in Caffe
            loss[name][it], acc[name][it] = (s.net.blobs[b].data.copy()
                                             for b in blobs)
        if it % disp_interval == 0 or it + 1 == niter:
            loss_disp = '; '.join('%s: loss=%.3f, acc=%2d%%' %
                                  (n, loss[n][it], np.round(100*acc[n][it]))
                                  for n, _ in solvers)
            print '%3d) %s' % (it, loss_disp)
    # Save the learned weights from both nets.
    weights = {}
    for name, s in solvers:
        filename = 'weights.%s.caffemodel' % name
        weights[name] = os.path.join(weight_dir, filename)
        s.net.save(weights[name])
    return loss, acc, weights
if __name__=='__main__':
    if (not os.path.exists(train_prototxt)):
       style_net(learn_all=True)
    if (not os.path.exists(val_prototxt)):
       style_net(train=False,learn_all=True)
    if (not os.path.exists(sovler_protxt_file)):
       style_net(train=False, learn_all=True)
    niter=200
    solver=caffe.get_solver(sovler_protxt_file)
    solver.net.copy_from(imageNet_weights)
    print 'Running solvers for %d iterations...' % niter
    loss, acc, weights=run_solvers(niter,[('style-net',solver)])
    print 'Done.'
    train_loss= loss['style-net']
    train_acc= acc['style-net']


