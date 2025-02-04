import torch
import torch.nn as nn
from .Transformers.DGCNN import DGCNN
from .Transformers.attentionMechanizm import PositionEmbedding, Encoder, Decoder, QueryGenerator

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.posembedding = PositionEmbedding(3, 128)
        self.dgcnn = DGCNN()

    def forward(self, x, teeth):
        le = self.dgcnn(x, teeth)
        pe = self.posembedding(x)
        return torch.cat([pe, le], dim = 2)

class Model(nn.Module):
    def __init__(self, inchannels=256, dk=256, dv=256, factor=4, num_heads=4, num_layers=2, offset=False):
        super(Model, self).__init__()
        self.features = FeatureExtractor()
        self.encoder = Encoder(inchannels=inchannels, dk=dk, dv=dv, factor=factor, num_heads=num_heads, num_layers=num_layers, offset=offset)

    def forward(self, x, teeth):
        x = self.features(x, teeth)
        return self.encoder(x)

class TransformerArchitecture(nn.Module):
    def __init__(self):
        super(TransformerArchitecture, self).__init__()
        self.encoder = Encoder(inchannels=256, dk=512, dv=512, factor=4, num_heads=4, num_layers=2)
        self.decoder = Decoder(inchannels=256, encoder_in=256, dk=512, dv=512, factor=4, num_heads=4, num_layers=2)
        self.QG = QueryGenerator(in_channels=256, num_points=1024, out_channels=256)

    def forward(self, x):
        encoder_in = self.encoder(x)
        query = self.QG(x)
        return self.decoder(query, encoder_in)
