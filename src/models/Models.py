import torch.nn as nn
import torch
from .HigherModels import *

from efficientnet_pytorch import EfficientNet
import torchvision


class ResNetAttention(nn.Module):
    def __init__(self, label_dim=527, pretrain=True):
        super(ResNetAttention, self).__init__()

        self.model = torchvision.models.resnet50(pretrained=False)

        if pretrain == False:
            print('ResNet50 Model Trained from Scratch (ImageNet Pretraining NOT Used).')
        else:
            print('Now Use ImageNet Pretrained ResNet50 Model.')

        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # remove the original ImageNet classification layers to save space.
        self.model.fc = torch.nn.Identity()
        self.model.avgpool = torch.nn.Identity()

        # attention pooling module
        self.attention = Attention(
            2048,
            label_dim,
            att_activation='sigmoid',
            cla_activation='sigmoid')
        self.avgpool = nn.AvgPool2d((4, 1))

    def forward(self, x):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        batch_size = x.shape[0]
        x = self.model(x)
        x = x.reshape([batch_size, 2048, 4, 33])
        x = self.avgpool(x)
        x = x.transpose(2,3)
        out, norm_att = self.attention(x)
        return out

class MBNet(nn.Module):
    def __init__(self, label_dim=527, pretrain=True):
        super(MBNet, self).__init__()

        self.model = torchvision.models.mobilenet_v2(pretrained=pretrain)

        self.model.features[0][0] = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.classifier = torch.nn.Linear(in_features=1280, out_features=label_dim, bias=True)

    def forward(self, x, nframes):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        out = torch.sigmoid(self.model(x))
        return out


class EffNetAttention(nn.Module):
    def __init__(self, label_dim=527, b=0, pretrain=True, head_num=4):
        super(EffNetAttention, self).__init__()
        self.middim = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]
        if pretrain == False:
            print('EfficientNet Model Trained from Scratch (ImageNet Pretraining NOT Used).')
            self.effnet = EfficientNet.from_name('efficientnet-b'+str(b), in_channels=1)
        else:
            print('Now Use ImageNet Pretrained EfficientNet-B{:d} Model.'.format(b))
            self.effnet = EfficientNet.from_pretrained('efficientnet-b'+str(b), in_channels=1)
        # multi-head attention pooling
        if head_num > 1:
            print('Model with {:d} attention heads'.format(head_num))
            self.attention = MHeadAttention(
                self.middim[b],
                label_dim,
                att_activation='sigmoid',
                cla_activation='sigmoid')
        # single-head attention pooling
        elif head_num == 1:
            print('Model with single attention heads')
            self.attention = Attention(
                self.middim[b],
                label_dim,
                att_activation='sigmoid',
                cla_activation='sigmoid')
        # mean pooling (no attention)
        elif head_num == 0:
            print('Model with mean pooling (NO Attention Heads)')
            self.attention = MeanPooling(
                self.middim[b],
                label_dim,
                att_activation='sigmoid',
                cla_activation='sigmoid')
        else:
            raise ValueError('Attention head must be integer >= 0, 0=mean pooling, 1=single-head attention, >1=multi-head attention.')

        self.avgpool = nn.AvgPool2d((4, 1))
        #remove the original ImageNet classification layers to save space.
        self.effnet._fc = nn.Identity()

    def forward(self, x, nframes=1056):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        print("Dimensions before extract_features:", x.size())
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        x = self.effnet.extract_features(x)
        print("Dimensions after extract_features:", x.size())
        x = self.avgpool(x)
        x = x.transpose(2,3)
        out, norm_att = self.attention(x)
        return out


#################################################################################################
import torch
import math
import torch.nn as nn
import torch.nn.functional as func
import numpy as np


# class for Front-End + Patches
class FrontEnd(nn.Module):

    def __init__(self, inputdim=400, output_dim=64, latent_dim=2048, dropout=0.2, wind_factor=1):
        super().__init__()
        self.fc1 = nn.Linear(in_features=inputdim, out_features=latent_dim)
        self.fc2 = nn.Linear(in_features=latent_dim, out_features=output_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.avg_pool = nn.AvgPool1d(kernel_size=wind_factor, stride=wind_factor)
        self.lambd = LambdaLayer(lambda x: x.transpose(1, 2))

        self.init_weights()
        # print(self.fc1.weight.dtype)

        # Information
        self.feed_forward_param_num = inputdim * output_dim * latent_dim

    def init_weights(self):
        initrange = 0.1
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()

    def forward(self, x):
        # print(x.dtype)
        x = self.lambd(x)
        x = self.avg_pool(x)
        #x = self.lambd(x)
        x = self.dropout(func.leaky_relu(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def transformer_block(dim_model, num_head, dim_feedforward, dropout,
                      num_encode_layers, dense_dim, pooling_kernel, stride):
    encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model,
                                               nhead=num_head,
                                               dim_feedforward=dim_feedforward,
                                               dropout=dropout,
                                               batch_first=True,
                                               norm_first=True)
    sequence = nn.Sequential(
        nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_encode_layers),
        nn.Linear(in_features=dim_model, out_features=dense_dim),
        nn.LeakyReLU(),
        nn.Dropout(p=dropout),
        nn.Linear(in_features=dense_dim, out_features=dense_dim),
        nn.LeakyReLU(),
        nn.Dropout(p=dropout),
        nn.Linear(in_features=dense_dim, out_features=dim_model),
        nn.LeakyReLU(),
        nn.Dropout(p=dropout),
        LambdaLayer(lambda x: x.transpose(1, 2)),
        nn.AvgPool1d(kernel_size=pooling_kernel, stride=stride),
        LambdaLayer(lambda x: x.transpose(1, 2)))
    return sequence


# defining a class for transformer
class BaseTransformer(nn.Module):
    """
    Initial Transformer class (16/12/22), from a guide to transformer architectures
    with some minor changes.
    """

    # constructor
    def __init__(
            self,
            dim_model=64,
            dropout=0.1,
            dense_dim=128,
            num_head=8,
            pooling_kernel_initial=2,
            pooling_kernel_last=32,
            num_encode_layers=2,
            dim_feedforward=2048,
            label_number=200,
            wind_factor=1
    ):
        super().__init__() #parent class constructor

        # Information
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # Layers
        self.positional_encoder = PositionalEncoding(d_model=dim_model, dropout=dropout)
        self.embedding = FrontEnd(inputdim=3000, output_dim=dim_model, dropout=dropout, wind_factor=wind_factor)
        self.dropout = nn.Dropout(p=dropout)
        self.output1 = nn.Linear(in_features=dim_model, out_features=dim_feedforward)
        self.output2 = nn.Linear(in_features=dim_feedforward, out_features=label_number)
        self.transformer_1 = transformer_block(dim_model=dim_model, num_head=num_head, dim_feedforward=dim_feedforward,
                                               dropout=dropout, num_encode_layers=num_encode_layers,
                                               dense_dim=dense_dim, pooling_kernel=pooling_kernel_initial, stride=2)
        self.transformer_2 = transformer_block(dim_model=dim_model, num_head=num_head, dim_feedforward=dim_feedforward,
                                               dropout=dropout, num_encode_layers=num_encode_layers,
                                               dense_dim=dense_dim, pooling_kernel=pooling_kernel_initial, stride=2)
        self.transformer_3 = transformer_block(dim_model=dim_model, num_head=num_head, dim_feedforward=dim_feedforward,
                                               dropout=dropout, num_encode_layers=num_encode_layers,
                                               dense_dim=dense_dim, pooling_kernel=pooling_kernel_last, stride=1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1

        # initializing weights for the transformer blocks
        transformer_blocks = [self.transformer_1, self.transformer_2, self.transformer_3]
        for block in transformer_blocks:
            if isinstance(block, nn.Linear):
                block.weight.data.uniform_(initrange, initrange)
                block.bias.data.zero_()

        # initializing weights fc layers
        # linear_layers = [self.latent1, self.latent2, self.output1, self.output2]
        linear_layers = [self.output1, self.output2]
        for layer in linear_layers:
            layer.weight.data.uniform_(-initrange, initrange)
            layer.bias.data.zero_()

    def forward(self, source):
        # source size = (Sequence Len, Batch Size)
        # Embedding phase
        source = self.embedding(source)

        # Positional Encoding
        source = self.positional_encoder(source)

        # blocks of transformer
        x1 = self.transformer_1(source)
        x1 = self.transformer_2(x1)
        x1 = self.transformer_3(x1)

        # output
        output = self.dropout(func.leaky_relu(self.output1(x1)))
        output = torch.sigmoid(self.output2(output))
        return output


class PositionalEncoding(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


##########################################################################################################
class TransformerEffNet(nn.Module):
    """
    Transformer with the efficient net as the frontend encoder
    """
    # constructor
    def __init__(
            self,
            impretrain,
            b,
            dim_model=128,
            dropout=0.2,
            dense_dim=128,
            num_head=8,
            pooling_kernel_initial=2,
            pooling_kernel_last=23,
            num_encode_layers=2,
            dim_feedforward=2048,
            label_number=200,
            effnet_dim=1408,
            wind_factor=1
    ):
        super().__init__() #parent class constructor

        # Information
        self.model_type = "Transformer"
        self.dim_model = dim_model
        self.b = b
        self.impretrain = impretrain

        # Layers
        self.positional_encoder = PositionalEncoding(d_model=dim_model, dropout=dropout)
        self.embedding = EffNetFrontEnd(b=self.b, pretrain=self.impretrain)
        self.dropout = nn.Dropout(p=dropout)
        self.dense_eff_to_transformer = nn.Linear(in_features=effnet_dim, out_features=dim_model)
        self.output1 = nn.Linear(in_features=dim_model, out_features=dim_feedforward)
        self.output2 = nn.Linear(in_features=dim_feedforward, out_features=label_number)
        self.transformer_1 = transformer_block(dim_model=dim_model, num_head=num_head, dim_feedforward=dim_feedforward,
                                               dropout=dropout, num_encode_layers=num_encode_layers,
                                               dense_dim=dense_dim, pooling_kernel=pooling_kernel_initial, stride=2)
        self.transformer_2 = transformer_block(dim_model=dim_model, num_head=num_head, dim_feedforward=dim_feedforward,
                                               dropout=dropout, num_encode_layers=num_encode_layers,
                                               dense_dim=dense_dim, pooling_kernel=pooling_kernel_initial, stride=2)
        self.transformer_3 = transformer_block(dim_model=dim_model, num_head=num_head, dim_feedforward=dim_feedforward,
                                               dropout=dropout, num_encode_layers=num_encode_layers,
                                               dense_dim=dense_dim, pooling_kernel=pooling_kernel_last, stride=1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1

        # initializing weights for the transformer blocks
        transformer_blocks = [self.transformer_1, self.transformer_2, self.transformer_3]
        for block in transformer_blocks:
            if isinstance(block, nn.Linear):
                block.weight.data.uniform_(initrange, initrange)
                block.bias.data.zero_()

        # initializing weights fc layers
        linear_layers = [self.output1, self.output2, self.dense_eff_to_transformer]
        for layer in linear_layers:
            layer.weight.data.uniform_(-initrange, initrange)
            layer.bias.data.zero_()

    def forward(self, source):

        # Positional Encoding
        source = self.positional_encoder(source)

        # Embedding phase
        source = self.embedding(source)
        source = self.dropout(func.leaky_relu(self.dense_eff_to_transformer(source)))

        # blocks of transformer
        x1 = self.transformer_1(source)
        #print("1T:", x1.size())
        x1 = self.transformer_2(x1)
        #print("2t:", x1.size())
        x1 = self.transformer_3(x1)
        #print("3t", x1.size())
        # output
        output = self.dropout(func.leaky_relu(self.output1(x1)))

        output = torch.sigmoid(self.output2(output))
        #print("4t", output.size())
        return output


class EffNetFrontEnd(nn.Module):
    def __init__(self, b=0, pretrain=True):
        super(EffNetFrontEnd, self).__init__()
        self.middim = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]
        if pretrain == False:
            print('EfficientNet Model Trained from Scratch (ImageNet Pretraining NOT Used).')
            self.effnet = EfficientNet.from_name('efficientnet-b'+str(b), in_channels=1)
        else:
            print('Now Use ImageNet Pretrained EfficientNet-B{:d} Model.'.format(b))
            self.effnet = EfficientNet.from_pretrained('efficientnet-b'+str(b), in_channels=1)
        self.avgpool = nn.AvgPool2d((4, 1))
        # remove the original ImageNet classification layers to save space.
        self.effnet._fc = nn.Identity()

    def forward(self, x):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        #print("1:", x.size())
        x = x.unsqueeze(1)
        #print("2:", x.size())
        x = x.transpose(2, 3)
        #print("3:", x.size())
        x = self.effnet.extract_features(x)
        x = self.avgpool(x).squeeze()
        #print("4:", x.size())
        x = x.transpose(1,2)
        #print("5:", x.size())
        return x


######################################################################################################

class EffNetAttentionConc(nn.Module):
    def __init__(self, label_dim=527, b=0, pretrain=True, head_num=4):
        super(EffNetAttentionConc, self).__init__()
        self.middim = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]
        if pretrain == False:
            print('EfficientNet Model Trained from Scratch (ImageNet Pretraining NOT Used).')
            self.effnet = EfficientNet.from_name('efficientnet-b'+str(b), in_channels=1)
        else:
            print('Now Use ImageNet Pretrained EfficientNet-B{:d} Model.'.format(b))
            self.effnet = EfficientNet.from_pretrained('efficientnet-b'+str(b), in_channels=1)
        # multi-head attention pooling
        if head_num > 1:
            print('Model with {:d} attention heads'.format(head_num))
            self.attention1 = MHeadAttention(
                self.middim[b],
                int(self.middim[b]/2),
                att_activation='sigmoid',
                cla_activation='sigmoid',
                head_num = 4,
                final_layer=False)
            self.attention2 = MHeadAttention(
                int(self.middim[b]/2),
                int(self.middim[b]/4),
                att_activation='sigmoid',
                cla_activation='sigmoid',
                head_num=2,
                final_layer=False)
            self.attention3 = MHeadAttention(
                int(self.middim[b] / 4),
                label_dim,
                att_activation='sigmoid',
                cla_activation='sigmoid',
                head_num = 1)
        # single-head attention pooling
        elif head_num == 1:
            print('Model with single attention heads')
            self.attention = Attention(
                self.middim[b],
                label_dim,
                att_activation='sigmoid',
                cla_activation='sigmoid')
        # mean pooling (no attention)
        elif head_num == 0:
            print('Model with mean pooling (NO Attention Heads)')
            self.attention = MeanPooling(
                self.middim[b],
                label_dim,
                att_activation='sigmoid',
                cla_activation='sigmoid')
        else:
            raise ValueError('Attention head must be integer >= 0, 0=mean pooling, 1=single-head attention, >1=multi-head attention.')

        self.avgpool = nn.AvgPool2d((4, 1))
        #remove the original ImageNet classification layers to save space.
        self.effnet._fc = nn.Identity()

    def forward(self, x, nframes=1056):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        x = self.effnet.extract_features(x)
        x = self.avgpool(x)
        x = x.transpose(2,3)
        out1, norm_att1 = self.attention1(x)
        out2, norm_att2 = self.attention2(out1)
        out, norm_att3 = self.attention3(out2)
        return out





##########################################################################################################
if __name__ == '__main__':
    input_tdim = 1056
    # ast_mdl = ResNetNewFullAttention(pretrain=False)
    # psla_mdl = EffNetFullAttention(pretrain=False, b=0, head_num=0)
    # input a batch of 10 spectrogram, each with 100 time frames and 128 frequency bins
    test_input = torch.rand([10, input_tdim, 128])
    test_output = psla_mdl(test_input)
    # output should be in shape [10, 527], i.e., 10 samples, each with prediction of 527 classes.
    print(test_output.shape)