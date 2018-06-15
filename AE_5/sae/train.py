# -*- coding: utf-8 -*-
'''
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import tuple_dataset
import numpy as np
import sys
sys.path.append("//tera/user/boku/study/nn")
import iomod as io
import argparse

class AE(chainer.Chain):
    def __init__(self,image_size,hidden1,hidden2):
        super(AE, self).__init__(
            l11 = L.Linear(image_size,hidden1),
            l12 = L.Linear(hidden1, hidden2),
            l13 = L.Linear(hidden2, hidden1),
            l14 = L.Linear(hidden1,image_size),
        )

    def __call__(self,x,):
        fv1 = F.sigmoid(self.l11(x))
        fv2 = F.sigmoid(self.l12(fv1))
        bv1 = F.sigmoid(self.l13(fv2))
        bv2 = F.sigmoid(self.l14(bv1))
        return bv2
    

def main():
    parser = argparse.ArgumentParser(description='AE:')
    #Required
    parser.add_argument('train', help='Train file path')
    parser.add_argument('output',type = str,  help='Output file path')

    #Option
    parser.add_argument('--image_side_size', '-image_size', type=int, default=9, help='image_side_size')
    parser.add_argument('--train_data_size', '-train_size', type=int, default=86375, help='train_data_size')
    #parser.add_argument('--test_data_size', '-test_size', type=int, default=500, help='test_data_size')
    parser.add_argument('--hidden1', '-h1', type=int, default=100, help='hidden1 layer size')
    parser.add_argument('--hidden2', '-h2', type=int, default=7, help='hidden2 layer size')
    parser.add_argument('--batchsize', '-b', type=int, default=1000, help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=1000,help='epoch size')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='0:GPU -1:CPU')
    parser.add_argument('--out', '-o', default='result',help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',help='Resume the training from snapshot')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('train data size: {}'.format(args.train_data_size))
    print('# hidden1: {}'.format(args.hidden1))
    print('# hidden2: {}'.format(args.hidden2))
    print('')
        
    #parameter
    image_size = args.image_side_size * args.image_side_size * args.image_side_size
    
    #load_file_train
    train = np.fromfile(args.train,np.float64)

    train = train.astype(np.float32)
    train = train.reshape(image_size, args.train_data_size)
    train_max = np.max(train,axis = 0)
    train_min = np.min(train,axis = 0)
    train_mat = ( train - train_min[np.newaxis,:] ) / ( train_max[np.newaxis,:] - train_min[np.newaxis,:])
    train_mat = train_mat.T
    print(train_mat.shape)

    #model
    model = L.Classifier(AE(image_size,args.hidden1,args.hidden2),lossfun = F.mean_squared_error)
    model.compute_accuracy = False
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    #train
    print ("train_AE")
    xtrain = tuple_dataset.TupleDataset(train_mat,train_mat)
    train_iter = chainer.iterators.SerialIterator(xtrain,args.batchsize)
    
    updater = chainer.training.StandardUpdater(train_iter, optimizer, device = args.gpu)
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), out='result')
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport( ['epoch', 'main/loss']))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.PlotReport(['main/loss'], 'epoch', file_name = args.output + 'AE_loss.png'))
    trainer.extend(extensions.ProgressBar())
    trainer.run()
    print
    #save_model 
    model.to_cpu()
    chainer.serializers.save_npz(args.output + 'my_AE.npz',model)

if __name__ == '__main__':
    main()
#F:\study_M1\input_data\gene_learn2.raw F:\study_M1\ae_5\

'''

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
import argparse

class AE(chainer.Chain):
    def __init__(self):
        super(AE, self).__init__(
            l11 = L.Linear(25,10),
            l12 = L.Linear(10,10),
            l13 = L.Linear(10,2),
        )

    def __call__(self,x,):
        h = F.sigmoid(self.l11(x))
        h = F.sigmoid(self.l12(h))
        return self.l13(h)
    

def main():
    parser = argparse.ArgumentParser(description='AE:')
    #Required
    parser.add_argument('train', help='Train file path')
    parser.add_argument('result',help='Train result file path')
    parser.add_argument('output',type = str,  help='Output file path')

    #Option
    parser.add_argument('--image_size', '-im_size', type=int, default=25, help='image_side_size')
    parser.add_argument('--train_data_size', '-train_size', type=int, default=5600, help='train_data_size')
    #parser.add_argument('--test_data_size', '-test_size', type=int, default=500, help='test_data_size')
    parser.add_argument('--hidden1', '-h1', type=int, default=100, help='hidden1 layer size')
    parser.add_argument('--hidden2', '-h2', type=int, default=7, help='hidden2 layer size')
    parser.add_argument('--batchsize', '-b', type=int, default=1000, help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=1000,help='epoch size')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='0:GPU -1:CPU')
    parser.add_argument('--out', '-o', default='result',help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',help='Resume the training from snapshot')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('train data size: {}'.format(args.train_data_size))
    print('# hidden1: {}'.format(args.hidden1))
    print('# hidden2: {}'.format(args.hidden2))
    print('')
        
    
    #load_file_train
    train = np.fromfile(args.train,np.float64)
    result = np.fromfile(args.result,np.float64)
    train = train.astype(np.float32)
    result = result.astype(np.int32)
    train = train.reshape(args.train_data_size,args.image_size)
    result = result.reshape(args.train_data_size,args.image_size)
    train_max = np.max(train,axis = 0)
    train_min = np.min(train,axis = 0)
    train_mat = ( train - train_min[np.newaxis,:] ) / ( train_max[np.newaxis,:] - train_min[np.newaxis,:])
    train_mat = train_mat
    result_mat = result

    print(train_mat.shape)
    print(train)
    print(result_mat.shape)
    print(result_mat)

    #model
    model = L.Classifier(AE())
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    #train
    print ("train_AE")
    xtrain = tuple_dataset.TupleDataset(train_mat,result_mat)
    train_iter = chainer.iterators.SerialIterator(xtrain,args.batchsize)
    updater = chainer.training.StandardUpdater(train_iter, optimizer, device = args.gpu)
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), out='result')
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport( ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())
    trainer.run()
    
    #save_model 
    model.to_cpu()
    chainer.serializers.save_npz(args.output + 'my_AE.npz',model)

if __name__ == '__main__':
    main()
