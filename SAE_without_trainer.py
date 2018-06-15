# -*- coding: utf-8 -*-
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable
from chainer import optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import sys
sys.path.append("//tera/user/boku/study/nn")
import iomod as io
import csv
import pickle
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

#
argvs = sys.argv
argc = len(argvs)
if (argc != 4):
    print ("Usage: Learn_path Test_path OUtput_path")
    quit()


#parameter
image_side_size = 9
image_size = image_side_size * image_side_size * image_side_size
learn_data_size = 1000
test_data_size = 500
hidden1  = 100
hidden2 = 3
pre_N = 1
N = 300

#load_file_learn
learn = np.fromfile(argvs[1],np.float64)
learn = learn.astype(np.float32)
learn = learn.reshape(image_size,learn_data_size)
learn_max = np.max(learn,axis = 0)
learn_min = np.min(learn,axis = 0)
learn_mat = ( learn - learn_min[np.newaxis,:] ) / ( learn_max[np.newaxis,:] - learn_min[np.newaxis,:])
xtrain = learn_mat.T

#load_file_test
test = np.fromfile(argvs[2],np.float64)
test = test.astype(np.float32)
test = test.reshape(image_size,test_data_size)
test_max = np.max(test,axis = 0)
test_min = np.min(test,axis = 0)
test_mat = ( test - test_min[np.newaxis,:] ) / ( test_max[np.newaxis,:] - test_min[np.newaxis,:])
xtest = test_mat.T

#input_test_save
trans_test = test.T
in_temp = trans_test.copy(order = 'C')
for t in range(test_data_size):
    io.save_raw(in_temp[t,:], argvs[3] + "sae/input_test" + str(t) +".raw",np.float32)



# Define model
class AE1(Chain):
    def __init__(self):
        super(AE1, self).__init__(
            l1=L.Linear(image_size,hidden1),
            l2=L.Linear(hidden1,image_size),
        )

    def __call__(self,x):
        bv = self.fwd(x)
        return F.mean_squared_error(bv, x)

    def fwd(self,x):
        fv = F.sigmoid(self.l1(x))
        bv = F.sigmoid(self.l2(fv))
        f1 = open(argvs[3] + "sae/picklep_sae_fv.dump", "wb")
        pickle.dump(fv, f1)
        f1.close()
        return bv

class AE2(Chain):
    def __init__(self):
        super(AE2, self).__init__(
            l3=L.Linear(hidden1,hidden2),
            l4=L.Linear(hidden2,hidden1),
        )

    def __call__(self,x):
        bv = self.fwd(x)
        return F.mean_squared_error(bv, x)

    def fwd(self,x):
        fv = F.sigmoid(self.l3(x))
        bv = F.sigmoid(self.l4(fv))
        return bv

class MyAE(Chain):
    def __init__(self):
        super(MyAE, self).__init__(
            l11=L.Linear(image_size,hidden1,initialW = model1.l1.W.data,initial_bias =  model1.l1.b.data),
            l12=L.Linear(hidden1,hidden2,initialW = model2.l3.W.data, initial_bias = model2.l3.b.data),
            l13 =L.Linear(hidden2,hidden1,initialW = model2.l4.W.data, initial_bias = model2.l4.b.data),
            l14 =L.Linear(hidden1,image_size,initialW = model1.l2.W.data, initial_bias = model1.l2.b.data),
        )

    def __call__(self,x):
        bv2 = self.fwd(x)
        return F.mean_squared_error(bv2, x)

    def fwd(self,x):
        fv1 = F.sigmoid(self.l11(x))
        fv2 = F.sigmoid(self.l12(fv1))
        bv1 = F.sigmoid(self.l13(fv2))
        bv2 = F.sigmoid(self.l14(bv1))
        return bv2

# Initialize model
model1 = AE1()
optimizer = optimizers.Adam()
optimizer.setup(model1)
train_losses = []
test_losses = []

# pre_training1
print ("pre_training1")
for i in range(pre_N):
    x_batch = Variable(xtrain)
    model1.zerograds()
    loss = model1(x_batch)
    loss.backward()
    optimizer.update()
#    print (loss.data)

f = open(argvs[3] + "sae/picklep_sae_fv.dump", "rb")
temp = pickle.load(f)
xtrain2 = temp.data
f.close()

#pre_training2
print ("pre_training2")
model2 = AE2()
optimizer.setup(model2)
for j in range(pre_N):
    x = Variable(xtrain2)
    model2.zerograds()
    loss = model2(x)
    loss.backward()
    optimizer.update()
#    print (loss.data)

#learn
print ("learn")
model3 = MyAE()
optimizer.setup(model3)
for i in range(N):
    x_batch = Variable(xtrain)
    model3.zerograds()
    train_loss = model3(x_batch)
    train_loss.backward()
    optimizer.update()
    train_losses.append(train_loss.data)
    print (train_loss.data)

    #test_loss
    x_batch = Variable(xtest)
    test_loss = model3(x_batch)
    test_losses.append(test_loss.data)
#    print (test_loss.data)

'''
#loss_save
print "train_loss"
for i in range(len(train_losses)):
    print '%f\n' % (train_losses[i]),
print '\n'

print "test_loss"
for i in range(len(test_losses)):
    print '%f\n' % (test_losses[i]),
print '\n'
'''

#final_result
x = Variable(xtest, volatile='on')
t1 = F.sigmoid(model3.l11(x))
t2 = F.sigmoid(model3.l12(t1))
with open(argvs[3] + 'sae/hidden_out.csv', 'wt') as f:
    writer = csv.writer(f)
    writer.writerows(t2.data)
t3 = F.sigmoid(model3.l13(t2))
y = F.sigmoid(model3.l14(t3))
print(y.shape)

temp_out = ( y.data.T * ( test_max[np.newaxis,:] - test_min[np.newaxis,:])) + test_min[np.newaxis,:]
temp_out2 = temp_out.T

for t in range(test_data_size):
    io.save_raw(temp_out2[t,:],argvs[3] + "sae/output_test" + str(t) + ".raw",np.float32)


print (test.shape)
print(test)
print (temp_out.shape)
print(temp_out)
tenmpp = abs(test - temp_out)
tenmpp2 = np.average(tenmpp,axis = 0)
gene =  np.average(tenmpp2)
with open(argvs[3] + 'sae/file_gene1.csv', 'wt') as f:
    writer = csv.writer(f)
    writer.writerows(tenmpp)
print (tenmpp2)
print ("gene")
print (gene)

'''
#knini_keizyo
data=np.loadtxt('hidden_in.csv',delimiter=',',dtype=np.float32)
print (data.shape)
hidden_out = Variable(data, volatile='on')
t3_hidden = F.sigmoid(model3.l13(hidden_out))
y_hidden = F.sigmoid(model3.l14(t3_hidden))
hidden_temp_out = y_hidden.data.T
#hidden_temp_out = (y_hidden.data.T* ( test_max[np.newaxis,:] - test_min[np.newaxis,:])) + test_min[np.newaxis,:]
hidden_temp_out2 = hidden_temp_out.T
for t in range(17):
    io.save_raw(hidden_temp_out2[t,:]*100,"C:/Users/yourb/Desktop/sae1/hiddenput_test" + str(t) + ".raw",np.float32)
'''

#matplotlib_setting
plt.plot(train_losses,'b',label = "train_error1")
plt.plot(test_losses,'r',label  = "test_error1")
plt.legend()
plt.grid()
plt.show()
