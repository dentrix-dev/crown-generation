import torch
import torch.nn as nn
from .Transformers.DGCNN import DGCNN
from .Transformers.attentionMechanizm import PositionEmbedding, Encoder, Decoder, QueryGenerator
from .FoldingNet.FoldingNet import FoldingNet

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.posembedding = PositionEmbedding(3, 128)
        self.dgcnn = DGCNN()

    def forward(self, x, teeth):
        le = self.dgcnn(x, teeth)
        pe = self.posembedding(x)
        return torch.cat([pe, le], dim = 2)

class TransformerArchitecture(nn.Module):
    def __init__(self):
        super(TransformerArchitecture, self).__init__()
        self.encoder = Encoder(inchannels=256, dk=512, dv=512, factor=4, num_heads=4, num_layers=2)
        self.decoder = Decoder(inchannels=256, encoder_in=256, dk=512, dv=512, factor=4, num_heads=4, num_layers=2)
        self.QG = QueryGenerator(in_channels=256, num_points=1024, out_channels=256)

    def forward(self, x):
        encoder_in = self.encoder(x)
        query, points = self.QG(x)
        return self.decoder(query, encoder_in), points

class Model(nn.Module):
    def __init__(self, inchannels=256, dk=256, dv=256, factor=4, num_heads=4, num_layers=2, num_points = 32, offset=False):
        super(Model, self).__init__()
        self.features = FeatureExtractor()
        self.transformer = TransformerArchitecture() # (inchannels=inchannels, dk=dk, dv=dv, factor=factor, num_heads=num_heads, num_layers=num_layers, offset=offset)
        self.projection = nn.Conv1d(in_channels=256+3, out_channels=3, kernel_size=1)
        self.foldingnet = FoldingNet(num_points=num_points)

    def forward(self, x, teeth):
        x = self.features(x, teeth)
        x, points = self.transformer(x)
        x = torch.cat([x, points], dim = 2)
        x = self.projection(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.foldingnet(x) + points
