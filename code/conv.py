import torch
import torch.nn as nn


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
        # TODO: Initialize the kernel
        if kernel_tensor == None:
            self.kernel = torch.randint(0, 2, (self.kernel_size, self.kernel_size))
        else:
            self.kernel = kernel_tensor
        # print("kernel :\n", self.kernel)
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
        # print('padded image: \n', padded_image)
        # print('padded image shape: ', padded_image.shape)
        target_shape = padded_image.shape[-1]-((self.kernel_size//2)+1), padded_image.shape[-2]-((self.kernel_size//2)+1)
        # print('target shape: ', target_shape)
        stop = (self.kernel_size//2) + 1

        conv_images = torch.zeros((image.shape[0], image.shape[1], target_shape[0], target_shape[1]))
        for n in range(padded_image.shape[0]):
            for c in range(padded_image.shape[1]):
                conv_image = list()
                this_padded_image = torch.squeeze(padded_image)
                # TODO: check the upper range bound for different kernel size
                for i in range(padded_image.shape[2]-stop):
                    for j in range(padded_image.shape[3]-stop):
                        # print(i+1, j+1)
                        this_padded_image_view = this_padded_image[i:i+self.kernel_size, j:j+self.kernel_size]
                        conv_image.append(torch.sum(this_padded_image_view * self.kernel).item())
                conv_image = torch.tensor(conv_image)
                conv_image = torch.reshape(conv_image, target_shape)
                # print(conv_image.shape)
                conv_image = torch.unsqueeze(conv_image, 0)
                # TODO: stack conv images
            conv_images[n] = conv_image
        # print(conv_images.shape)
        return conv_images


                        

    # def backward(self):
        """
        Backward pass 
        (leave the function commented out,
        so that autograd will be used.
        No action needed here.)
        :return:
        """



if __name__ == "__main__":
    image = torch.tensor([[1,1,1,0,0],[0,1,1,1,0],[0,0,1,1,1],[0,0,1,1,0],[0,1,1,0,0]])
    # k = torch.tensor([[1, 0, 1],[0, 1, 0],[1, 0, 1]])
    # image = image[:3, :3]
    # print(image.shape)
    # print(image * k)
    image = torch.unsqueeze(image, 0)
    image = torch.unsqueeze(image, 0)
    filter = torch.tensor([[1, 0, 1],[0, 1, 0],[1, 0, 1]])
    # print(image.shape)
    conv = Conv(3, kernel_tensor=filter)
    print(conv(image))
