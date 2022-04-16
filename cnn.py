import numpy as np
import torch 
import torch.nn as nn

class CNN(nn.Module):
    ''' Convolutional half module of Actor network to predict actions from state '''

    def __init__(self, image_dim):
        '''
        Input:
            128x128 grayscale image
        '''
        super().__init__()
        self.input_shape = image_dim
        self.conv_out_channels = 16
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, self.conv_out_channels, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.AvgPool2d(kernel_size=2)
        )

        # self.output_size = self.compute_output_size(self.input_shape)

    def forward(self, x):
        x = self.preprocess(x)
        return self.conv(x).flatten()

    def compute_output_size(self, img_shape) :
        
        # print(img_shape)
        x = torch.zeros(img_shape, dtype=torch.float32).permute((2, 0, 1)).unsqueeze(0)
        x = self.preprocess(x)
        print(x.shape)

        out = self.conv(x).flatten()
        return out.shape[0]

    def preprocess(self, obs):
        '''preprocess a batch of atari images to be input to the convolutional neural network
        '''
        #convert to grayscale
        obs = np.dot(obs[...,:3], [0.2989, 0.5870, 0.1140])
        #downsample by factor of 2
        obs = obs[..., ::2, ::2]
        #convert to tensor, add back batch dimension
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(1)
        
        return obs


if __name__ == '__main__':
    obs = torch.randn(2, 128, 128, 3)

    cnn = CNN((128, 128, 3))

    print (cnn.preprocess(obs))

