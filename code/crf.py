import torch
import torch.nn as nn
import crf_utils

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
        self.params = nn.Parameter(torch.zeros(num_labels * embed_dim + num_labels**2))
        self.w = params.narrow(0,0,num_labels * embed_dim).view(num_labels, embed_dim)
        self.T = params.narrow(0,num_labels * embed_dim, num_labels**2).view(num_labels,num_labels)
        ### Use GPU if available
        if self.use_cuda:
            [m.cuda() for m in self.modules()]

    @staticmethod
    def forward(self, X):
        """
        Implement the objective of CRF here.
        The input (features) to the CRF module should be convolution features.
        """
        self.features = nn.Parameter(self.get_conv_features(X))

        prediction = 
        return (prediction)

    def loss(self, X, labels):
        """
        Compute the negative conditional log-likelihood of a labelling given a sequence.
        """
        print(X.shape)
        features = nn.Parameter(conv(X))
        loss = crf_utils.obj_func(features, labels, params, C, num_labels, embed_dim)
        return loss

    # @staticmethod
    # def backward(self):
    #     """
    #     Return the gradient of the CRF layer
    #     :return:
    #     """
    #     gradient = crf_utils.crfFuncGrad(params, X, C, num_labels, embed_dim)
    #     return gradient

    def get_conv_features(self, X):
        """
        Generate convolution features for a given word
        """
        convfeatures = 
        return convfeatures

    