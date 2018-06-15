# -*- coding: utf-8 -*-
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import tuple_dataset
import numpy as np
import sys
sys.path.append("//tera/user/boku/study/nn")
import iomod as io
import csv

class AE_test(chainer.Chain):
    def __init__(self,image_size,hidden1,hidden2):
        super(AE_test, self).__init__(
            l11 = L.Linear(image_size,hidden1),
            l12 = L.Linear(hidden1, hidden2),
            l13 = L.Linear(hidden2, hidden1),
            l14 = L.Linear(hidden1,image_size),
        )

    def __call__(self,x):
        fv1 = F.sigmoid(self.l11(x))
        fv2 = F.sigmoid(self.l12(fv1))
        bv1 = F.sigmoid(self.l13(fv2))
        bv2 = F.sigmoid(self.l14(bv1))
        return bv2


#argument
argvs = sys.argv
argc = len(argvs)
if (argc != 4):
    print ("Usage: Model_file_path Test_file_path Output_file_path")
    quit()

#parameter
image_side_size = 9
image_size = image_side_size * image_side_size * image_side_size
test_data_size = 1340
hidden1  = 100
hidden2 = 7

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
    io.save_raw(in_temp[t,:], argvs[3] + "result_AE\\input_test" + str(t) +".raw",np.float32)

#load_model
model = L.Classifier(AE_test(image_size,hidden1,hidden2))
chainer.serializers.load_npz(argvs[1] ,model)

temp = AE_test(image_size,hidden1,hidden2)

test_result = model.predictor(np.array(test_mat)).data
test_result = ( test_result.T * ( test_max[np.newaxis,:] - test_min[np.newaxis,:])) + test_min[np.newaxis,:]
test_result_temp = test_result.T

#save_output_test
for t in range(test_data_size):
    io.save_raw(test_result_temp[t,:], argvs[3] + "result_AE\\output_test" + str(t) + ".raw",np.float32)

#Generalization
print ("Result_AE")
print (test.shape)
print(test)
print(test_result.shape)
print(test_result)
dev_result = abs(test - test_result)
gene =  np.average(np.average(dev_result,axis = 0))
print ("Generalization_AE")
print (gene)
