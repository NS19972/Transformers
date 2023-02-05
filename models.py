# Файл с самописными нейросетевыми моделями

import torch
import torch.nn as nn
import numpy as np
from custom_transformer_hyperparameters import *
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer
import math


class PositionalEncoder(nn.Module):
    def __init__(self, dropout: float = 0.1, max_seq_len: int = 5000, d_model: int = 512, batch_first: bool = True):
        super().__init__()
        d_model = int(np.ceil(d_model/2) * 2)
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        self.x_dim = -1 if batch_first else 0
        position = torch.arange(max_seq_len//2).unsqueeze(1)
        denominator = torch.exp(torch.arange(0, d_model) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(batch_size, max_seq_len, d_model)
        pe[:, 0::2, :] = torch.sin(position / denominator)
        pe[:, 1::2, :] = torch.cos(position / denominator)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[..., :x.size(self.x_dim)]
        return self.dropout(x)


class CustomTransformerEncoder(nn.Module):
    def __init__(self, input_shape, d_model, nhead, num_layers, **kwargs):
        super().__init__()
        self.positional_encoder = PositionalEncoder(max_seq_len=encoder_length)
        self.dimension_shift = nn.Linear(input_shape, d_model)
        self.relu = nn.ReLU()
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(d_model, nhead=nhead, batch_first=True, **kwargs)
                                            for i in range(num_layers)])
        self.norm = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x, mask=None):
        x = self.positional_encoder(x)
        x = self.relu(self.dimension_shift(x))
        for i in range(len(self.encoder_layers)):
            x = self.encoder_layers[i](x, src_mask=mask)
        x = self.norm(x)
        return x


class CustomTransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, **kwargs):
        super().__init__()
        self.positional_encoder = PositionalEncoder(max_seq_len=prediction_length)
        self.resize = nn.Linear(1, d_model)
        self.relu = nn.ReLU()
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(d_model, nhead=nhead, batch_first=True, **kwargs)
                                            for i in range(num_layers)])
        self.norm = nn.LayerNorm(normalized_shape=d_model)

    def forward(self, target, encoder_outputs, source_mask, target_mask):
        x = self.positional_encoder(target)
        x = self.relu(self.resize(x))
        for i in range(len(self.decoder_layers)):
            x = self.decoder_layers[i](x, encoder_outputs, source_mask, target_mask)
        x = self.norm(x)
        return x


class Transformer(nn.Module):
    def __init__(self, data, categorical_columns, real_columns, target_size, embedding_dim=64):
        super().__init__()
        num_real_columns = len(real_columns)
        self.categorical_embedding_layers = nn.ModuleList([nn.Embedding(num_embeddings=data[column].nunique(),
                                                          embedding_dim=min(embedding_dim, data[column].nunique()))
                                                          for column in categorical_columns])

        pos_enc_input_dim = sum([layer.embedding_dim for layer in self.categorical_embedding_layers]) + embedding_dim

        self.real_embedding_layer = nn.Linear(in_features=5+num_real_columns, out_features=embedding_dim)

        self.encoder = CustomTransformerEncoder(input_shape=pos_enc_input_dim,
                                                d_model=transformer_size,
                                                dim_feedforward=transformer_hidden_dim,
                                                nhead=attention_heads,
                                                num_layers=num_encoder_layers)

        self.decoder = CustomTransformerDecoder(d_model=transformer_size,
                                                dim_feedforward=transformer_hidden_dim,
                                                nhead=attention_heads,
                                                num_layers=num_decoder_layers)

        self.flatten_layer = nn.Flatten()
        self.output_layer = nn.Linear(prediction_length * transformer_size, target_size)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def get_embeddings(self, x_cat, x_cont):
        categorical_embeddings = []
        for i in range(len(self.categorical_embedding_layers)):
            categorical_embeddings.append(self.categorical_embedding_layers[i](x_cat[..., i]))

        real_embedding = self.real_embedding_layer(x_cont)
        embeddings = torch.cat(categorical_embeddings + [real_embedding], dim=-1)
        return embeddings

    def forward(self, x, src_mask=None, target_mask=None):
        x_cat, x_real, target = x['encoder_cat'], x['encoder_cont'], x['decoder_target']
        target = target.unsqueeze(-1)
        x = self.get_embeddings(x_cat, x_real)
        encoder_outputs = self.encoder(x)
        decoder_outputs = self.decoder(target, encoder_outputs, src_mask, target_mask)
        flattened_decoder_outputs = self.flatten_layer(decoder_outputs)
        output = self.output_layer(flattened_decoder_outputs)
        return output


# Not Used
class TimeSeriesDataloader:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        pass
