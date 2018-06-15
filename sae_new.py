# -*- coding: utf-8 -*-
import chainer
import chainer
from chainer import report, training, Chain, datasets, iterators, optimizers
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import tuple_dataset
import numpy as np
import sys
sys.path.append("//tera/user/boku/study/nn")
import iomod as io
import csv
import pickle
import matplotlib.pyplot as plt

#argument
argvs = sys.argv
argc = len(argvs)
if (argc != 4):
    print ("Usage: train_path Test_path OUtput_path")
    quit()


#parameter
image_side_size = 9
image_size = image_side_size * image_side_size * image_side_size
train_data_size = 1000
test_data_size = 500
hidden1  = 100
hidden2 = 3
n_epoch = 5

#load_file_train
train = np.fromfile(argvs[1],np.float64)
train = train.astype(np.float32)
train = train.reshape(image_size,train_data_size)
train_max = np.max(train,axis = 0)
train_min = np.min(train,axis = 0)
train_mat = ( train - train_min[np.newaxis,:] ) / ( train_max[np.newaxis,:] - train_min[np.newaxis,:])
train_mat = train_mat.T

#load_file_test
test = np.fromfile(argvs[2],np.float64)
test = test.astype(np.float32)
test = test.reshape(image_size,test_data_size)
test_max = np.max(test,axis = 0)
test_min = np.min(test,axis = 0)
test_mat = ( test - test_min[np.newaxis,:] ) / ( test_max[np.newaxis,:] - test_min[np.newaxis,:])
test_mat = test_mat.T

#save_input_test
trans_test = test.T
in_temp = trans_test.copy(order = 'C')
for t in range(test_data_size):
    io.save_raw(in_temp[t,:], argvs[3] + "sae/input_test" + str(t) +".raw",np.float32)


class MyAE(Chain):
    def __init__(self):
        super(MyAE, self).__init__(
            l11=L.Linear(image_size,hidden1),
            l12=L.Linear(hidden1,hidden2),
            l13 =L.Linear(hidden2,hidden1),
            l14 =L.Linear(hidden1,image_size),
        )

    def __call__(self,x,train = True):
        fv1 = F.sigmoid(self.l11(x))
        fv2 = F.sigmoid(self.l12(fv1))
        bv1 = F.sigmoid(self.l13(fv2))
        bv2 = F.sigmoid(self.l14(bv1))
        return bv2

#model
model = L.Classifier(MyAE(),lossfun = F.mean_squared_error)
model.compute_accuracy = False
optimizer = optimizers.Adam()
optimizer.setup(model)

#train
print ("train")
xtrain = tuple_dataset.TupleDataset(train_mat,train_mat)
xtest = tuple_dataset.TupleDataset(test_mat,test_mat)
train_iter = iterators.SerialIterator(xtrain,10)
test_iter = iterators.SerialIterator(xtest,1,repeat = False,shuffle = False)

updater = training.StandardUpdater(train_iter, optimizer)
trainer = training.Trainer(updater, (n_epoch, 'epoch'), out='result')

trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport( ['epoch', 'main/loss']))
trainer.extend(extensions.snapshot(), trigger=(n_epoch, 'epoch'))
trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(extensions.ProgressBar())
trainer.run()

#test
test_result = model.predictor(np.array(test_mat))
test_result2 = ( test_result.data.T * ( test_max[np.newaxis,:] - test_min[np.newaxis,:])) + test_min[np.newaxis,:]
test_result_temp = test_result2.T

#save_output_test
for t in range(test_data_size):
    io.save_raw(test_result_temp[t,:], argvs[3] + "sae/output_test" + str(t) + ".raw",np.float32)

#Generalization
dev_result = abs(test - test_result2)
gene =  np.average(np.average(dev_result,axis = 0))
with open(argvs[3] + 'sae/generalization.csv', 'wt') as f:
    writer = csv.writer(f)
    writer.writerows(dev_result)
print ("Generalization")
print (gene)
