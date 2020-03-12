import torch
import torch.nn as nn
import crf_utils
from conv import Conv

class CRF(nn.Module):

    def __init__(self, input_dim, embed_dim, conv_layers, num_labels, batch_size):
        """
        Linear chain CRF as in Assignment 2
        """
        super(CRF, self).__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.conv_layers = conv_layers
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.use_cuda = torch.cuda.is_available()
        """
        Initialize trainable parameters of CRF here
        """
        self.conv_layer = Conv(5)
        self.params = nn.Parameter(torch.zeros(num_labels * embed_dim + num_labels**2))
        # self.w = params.narrow(0,0,num_labels * embed_dim).view(num_labels, embed_dim)
        # self.T = params.narrow(0,num_labels * embed_dim, num_labels**2).view(num_labels,num_labels)
        ### Use GPU if available
        if self.use_cuda:
            [m.cuda() for m in self.modules()]

    def forward(self, X):
        """
        Implement the objective of CRF here.
        The input (features) to the CRF module should be convolution features.
        """
        X = self.__reshape_before_conv__(X)
        features = nn.Parameter(self.conv_layer(X))
        features = self.__reshape_after_conv__(features)

        prediction = crf_utils.dp_infer(features, self.params, self.num_labels, self.embed_dim)
        return (prediction)

    def loss(self, X, labels):      # Accepts Batches
        """
        Compute the negative conditional log-likelihood of a labelling given a sequence.
        """
        X = self.__reshape_before_conv__(X)
        features = nn.Parameter(self.conv_layer(X))
        features = self.__reshape_after_conv__(features)
        C = 1000
        loss = crf_utils.obj_func(features, labels, self.params, C, self.num_labels, self.embed_dim)
        return loss


    def __reshape_before_conv__(self, X):
        X = torch.reshape(X, (X.shape[0]*X.shape[1], 1, 16, 8))
        return X


    def __reshape_after_conv__(self, X):
        X = torch.reshape(X, (X.shape[0]//14, 14, X.shape[2]*X.shape[3]))
        return X


    # @staticmethod
    # def backward(self):
    #     """
    #     Return the gradient of the CRF layer
    #     :return:
    #     """
    #     gradient = crf_utils.crfFuncGrad(params, X, C, num_labels, embed_dim)
    #     return gradient

    # def get_conv_features(self, X):
    #     """
    #     Generate convolution features for a given word
    #     """
    #     # convfeatures = 
    #     return convfeatures

    
"""
# import torch
import torch.optim as optim
import torch.utils.data as data_utils
from data_loader import get_dataset
import numpy as np
# from crf import CRF


# Tunable parameters
batch_size = 64
num_epochs = 10
max_iters  = 1000
print_iter = 25 # Prints results every n iterations
conv_shapes = [[1,64,128]] #


# Model parameters
input_dim = 128
embed_dim = 128
num_labels = 26
cuda = torch.cuda.is_available()

# Instantiate the CRF model
crf_model = CRF(input_dim, embed_dim, conv_shapes, num_labels, batch_size)

# Setup the optimizer
opt = optim.LBFGS(crf_model.parameters())


##################################################
# Begin training
##################################################
step = 0

# Fetch dataset
dataset = get_dataset()
split = int(0.5 * len(dataset.data)) # train-test split
train_data, test_data = dataset.data[:split], dataset.data[split:]
train_target, test_target = dataset.target[:split], dataset.target[split:]

# Convert dataset into torch tensors
train = data_utils.TensorDataset(torch.tensor(train_data).float(), torch.tensor(train_target).long())
test = data_utils.TensorDataset(torch.tensor(test_data).float(), torch.tensor(test_target).long())

# print(len(train[0][1][0]))

# for i in range(num_epochs):
#     print("Processing epoch {}".format(i))
    
#     # Define train and test loaders
train_loader = data_utils.DataLoader(train,  # dataset to load from
                                            batch_size=batch_size,  # examples per batch (default: 1)
                                            shuffle=True,
                                            sampler=None,  # if a sampling method is specified, `shuffle` must be False
                                            num_workers=5,  # subprocesses to use for sampling
                                            pin_memory=False,  # whether to return an item pinned to GPU
                                            )

test_loader = data_utils.DataLoader(test,  # dataset to load from
                                        batch_size=batch_size,  # examples per batch (default: 1)
                                        shuffle=False,
                                        sampler=None,  # if a sampling method is specified, `shuffle` must be False
                                        num_workers=5,  # subprocesses to use for sampling
                                        pin_memory=False,  # whether to return an item pinned to GPU
                                        )
print('Loaded dataset... ')
for i_batch, sample in enumerate(train_loader):

    train_X = sample[0]
    train_Y = sample[1]

    # print(i_batch,len(train_X),len(train_X[0]),len(train_X[0][0]))
    print(i_batch,len(train_Y),len(train_Y[0]),len(train_Y[0][0]))
    # yt = train_Y[0]
    # print("numder of letters in the word = ",torch.sum(train_Y[0]), torch.sum(train_Y[1])) 
    # print(torch.all(train_Y[0].eq(train_Y[1])))
    # c = (~torch.eq(train_Y[0],train_Y[1])).sum()
    # print(c)
    # print(train_Y[0] - train_Y[1])
    
"""    
