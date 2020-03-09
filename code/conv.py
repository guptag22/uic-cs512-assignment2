import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    """
    Convolution layer.
    """
    def __init__(self, kernel_size, stride=1, padding=True, kernel_tensor=None):
        super(Conv, self).__init__()
        self.init_params(kernel_size, stride, padding, kernel_tensor)


    def init_params(self, kernel_size, stride, padding, kernel_tensor):
        """
        Initialize the layer parameters
        :return:
        """
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if kernel_tensor is None:
            # TODO: check if kernel is to be initialized from a distribution
            self.kernel = nn.Parameter(torch.randint(0, 2, (self.kernel_size, self.kernel_size), dtype=torch.float))
        else:
            self.kernel = nn.Parameter(kernel_tensor)
        self.padding_layer = nn.ZeroPad2d(self.kernel_size//2)

    def forward(self, image):
        """
        Forward pass
        :return:
        """
        if self.padding:
            padded_image = self.padding_layer(image)
        else:
            padded_image = image
        target_shape = padded_image.shape[-1]-((self.kernel_size//2)+1), padded_image.shape[-2]-((self.kernel_size//2)+1)
        stop = (self.kernel_size//2) + 1

        conv_images = torch.zeros((image.shape[0], image.shape[1], target_shape[0], target_shape[1]))
        for n in range(padded_image.shape[0]):
            for c in range(padded_image.shape[1]):
                conv_image = list()
                this_padded_image = torch.squeeze(padded_image)
                # TODO: check the upper range bound for different kernel size
                for i in range(padded_image.shape[2]-stop):
                    for j in range(padded_image.shape[3]-stop):
                        this_padded_image_view = this_padded_image[i:i+self.kernel_size, j:j+self.kernel_size]
                        conv_image.append(torch.sum(torch.mul(this_padded_image_view, self.kernel)))
                conv_image = torch.stack(conv_image)
                conv_image = torch.reshape(conv_image, target_shape)
                conv_image = torch.unsqueeze(conv_image, 0)
            conv_images[n] = conv_image
        return conv_images
                        

    # def backward(self):
        """
        Backward pass 
        (leave the function commented out,
        so that autograd will be used.
        No action needed here.)
        :return:
        """


def get_torch_conv(image, kernel, padding=0):
    return F.conv2d(image, kernel, padding=padding)


if __name__ == "__main__":
    image = torch.tensor([[1,1,1,0,0],[0,1,1,1,0],[0,0,1,1,1],[0,0,1,1,0],[0,1,1,0,0]], dtype=torch.float)
    image = torch.unsqueeze(image, 0)
    image = torch.unsqueeze(image, 0)
    kernel = torch.tensor([[1, 0, 1],[0, 1, 0],[1, 0, 1]], dtype=torch.float)
    conv = Conv(3, padding=True, kernel_tensor=kernel)
    print(conv(image))

    # check with PyTorch implementation
    kernel = torch.unsqueeze(kernel, 0)
    kernel = torch.unsqueeze(kernel, 0)
    print(get_torch_conv(image, kernel, 1))
