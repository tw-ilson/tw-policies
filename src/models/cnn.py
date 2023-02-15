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
        
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

        self.conv = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.AvgPool2d(kernel_size=2)
        )

        self.to(self.device)

        self.output_size = self.compute_output_size(self.input_shape)

    def forward(self, x):
        x = self.preprocess(x).to(self.device)
        return self.conv(x).flatten(1)

    def compute_output_size(self, img_shape) :
        
        x = torch.zeros(img_shape, dtype=torch.float32, device='cpu').unsqueeze(0)

        y = self.forward(x)
        return y.shape[1]

    def preprocess(self, obs):
        '''preprocess a batch of atari images to be input to the convolutional neural network
        '''

        if len(obs.shape) == 4:
            #convert to grayscale
            obs = np.dot(obs[...,:3], [0.2989, 0.5870, 0.1140])

        #downsample by factor of 2
        obs = obs[..., ::2, ::2]

        #convert to tensor, add back channel dimension
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(1)
        return obs

if __name__ == '__main__':
    obs = torch.randn(1, 180, 210, 3)

    cnn = CNN((180, 210, 3))

    print(cnn.preprocess(obs).shape)
    print (cnn.forward(obs).shape)
    # print(cnn.output_size)

