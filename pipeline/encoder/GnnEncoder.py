from typing import Union, Any, Iterable, Dict, List, Optional, Tuple

import torch
import torch_geometric as gtc
import torch_geometric.nn as gnn
from argparse import ArgumentParser

if __name__ == '__main__':
    from EncoderInterface import Encoder
else:
    from pipeline.encoder.EncoderInterface import Encoder

GNN_LAYERS = {
    'RGCNConv': gnn.RGCNConv,  # slower, but memory efficient
    'FastRGCNConv': gnn.FastRGCNConv,  # faster, but needs more memory

    'GCNConv': gnn.GCNConv,
    'GATConv': gnn.GATConv,
}

ACTIVATIONS = {
    'ReLU': torch.nn.ReLU(inplace = False),
    'LeakyReLU': torch.nn.LeakyReLU(inplace = False),
}

POOLINGS = {
    'global_add_pool': gnn.global_add_pool,
    'global_mean_pool': gnn.global_mean_pool
}

class GnnEncoder(Encoder):
    GNN_parser = ArgumentParser(parents=[Encoder.encoder_parser])
    GNN_parser.add_argument("--gnn_layer", type=str, required=False, default='GCNConv', help="GNN layer to be used")
    GNN_parser.add_argument("--num_layers", type=int, required=False, default=4, help="Number of GNN layers")
    GNN_parser.add_argument("--in_channels", type=int, required=False, default=768, help="Dimension of input, i.e. number on in-channels to first layer")
    GNN_parser.add_argument("--hidden_channels", type=int, required=False, default=32, help="Dimension of intermediate representations, i.e. number of in- and out-channels of intermediate layers")
    GNN_parser.add_argument("--out_channels", type=int, required=False, default=14, help="Dimension of output, i.e. number of out-channels of last layer. Setting `predict_number_of_frames` to True increases out_channles by 1 in GnnEncoder.")
    GNN_parser.add_argument("--num_relations", type=int, required=False, default=None, help="for RGCNConv: number of relations in the graph.")
    GNN_parser.add_argument("--heads", type=int, required=False, default=None, help="for GATConv: number of attention heads.")
    GNN_parser.add_argument("--activation", type=str, required=False, default='ReLU', help="activation (i.e. non-linearity) that is used after each layer")
    GNN_parser.add_argument("--pool", type=str, required=False, default='global_add_pool', help="pooling function to be used for aggregation of node features")
    GNN_parser.add_argument("--enable_dropout", action="store_true", default=False, required=False)

    def __init__(
        self, 
        gnn_layer: str = 'GCNConv', 
        num_layers: int = 4, 
        in_channels: int = 768,
        hidden_channels: int = 32, 
        out_channels: int = 14,
        num_relations: Optional[int] = None, 
        heads: Optional[int] = None, 
        activation: str = 'ReLU',
        pool: str = 'global_add_pool',
        enable_dropout: bool = False,
        predict_number_of_frames: bool = False,
    ) -> None:
        """
        Initializing an Decoder module (may be more than one module running in parallel)
         (Input-->Dataset-->ENCODER (should produce probabilities)-->aggregator-->final output)
        :param gnn_layer: what GNN layer is used
        :param num_layers: number of GNN layers
        :param in_channels: dimension of input, i.e. number on in-channels to first layer
        :param hidden_channels: dimension of intermediate representations, i.e. number of in- and out-channels of intermediate layers
        :param out_channels: dimension of representations after final graph layer, i.e. number of out-channels in final graph layers
        :param num_relations: for RGCNConv: number of relations in the graph.
        :param heads: for GATonv: number of attention heads.
        :param activation: activation (i.e. non-linearity) that is used after each layer
        :param pool: pooling function to be used for aggregation of node features
        :param enable_dropout: should dropout be used or not?
        :param predict_number_of_frames: should the expected number of frames also be computed or not? Setting this to True increases the output dimension by 1. 
        """
        super().__init__(
            predict_number_of_frames=predict_number_of_frames
        )
        self.gnn_layer_str = gnn_layer
        self.gnn_layer = GNN_LAYERS[gnn_layer]
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels + int(predict_number_of_frames)
        self.num_relations = num_relations
        self.heads = heads
        self.activation = ACTIVATIONS[activation]
        self.pool = POOLINGS[pool]

        self.gnn_layer_kwargs = {}
        self.gnn_final_layer_kwargs = {}
        if self.num_relations != None:
            self.gnn_layer_kwargs['num_relations'] = self.num_relations
            self.gnn_final_layer_kwargs['num_relations'] = self.num_relations
        if self.heads != None:
            self.gnn_layer_kwargs['heads'] = self.heads
            self.gnn_final_layer_kwargs['heads'] = 1

        # create layers
        assert self.num_layers >= 2, f'Current implementation does not support 1 layer.\n{self.num_layers = }'

        self.first_layer = self.gnn_layer(
            in_channels = self.in_channels,
            out_channels = self.hidden_channels,
            **self.gnn_layer_kwargs
        )

        self.intermediate_layers = torch.nn.ModuleList(
            [
                self.gnn_layer(
                    in_channels = self.hidden_channels * (self.heads if self.heads != None else 1),
                    out_channels = self.hidden_channels,
                    **self.gnn_layer_kwargs
                )
                for _ in range(self.num_layers - 2)
            ]
        )

        self.final_layer = self.gnn_layer(
            in_channels = self.hidden_channels * (self.heads if self.heads != None else 1),
            out_channels = self.out_channels,
            **self.gnn_final_layer_kwargs
        )

        if enable_dropout:
            self.dropout = torch.nn.Dropout(p=.25)
        else:
            self.dropout = torch.nn.Identity()

        if torch.cuda.is_available():
            self.to("cuda")

    def _one_layer(
        self, 
        layer: gnn.MessagePassing,
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_type: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Performs one layer of the GNN
        :param layer: layer to be applied
        :param x: node features for torch_geometric
        :param edge_index: edge_index for torch_geometric 
        :param edge_type: edge type for RGCN
        :return: node features after layer application
        """
        if self.gnn_layer_str in ['GCNConv', 'GATConv']:
            return layer(x, edge_index)
        
        if self.gnn_layer_str in ['RGCNConv', 'FastRGCNConv']:
            return layer(x, edge_index, edge_type)
        
        raise NotImplementedError(self.gnn_layer_str)

    def forward(
        self, 
        x: List[Dict[str, gtc.data.Data]]
    ) -> torch.Tensor:
        """
        Encodes the raw machine-readable input by predicting frame probabilities
        :param x: node features for torch_geometric
        # :param edge_index: edge_index for torch_geometric 
        # :param batch: batch for torch_geometric pooling funtions. Denotes which node belongs to which batch. 
        # :param kwargs: further (optional) arguments for forward pass of gnn layer
        :return: predictions in a shape of (batch, frames)
        (if predict_number_of_frames, each frame prediction has in the first
        place an estimation how many frames occur in the text)
        """
        x = [y['data'] for y in x]
        batch = gtc.data.Batch.from_data_list(x)
        y = self.dropout(batch.x)
        edge_index = batch.edge_index

        y = self._one_layer(self.first_layer, y, edge_index, batch.edge_type)
        
        y = self.activation(y)

        for l in self.intermediate_layers:
            y = self._one_layer(l, y, edge_index, batch.edge_type)
            y = self.activation(y)

        y = self._one_layer(self.final_layer, y, edge_index, batch.edge_type)
        
        y = self.pool(y, batch=batch.batch)  # graph representation

        y = self.sigmoid_outputs(y)  # frame probabilities

        return y

if __name__ == '__main__':
    import os
    import pandas as pd
    import sys
    import numpy as np

    sys.path.append('/home/mitarb/plenz/ws/projects/git_shared/SemEval233Framing')
    from pipeline.preprocessing.FrameDatasetGraph import FrameDatasetForGnn

    # init dataset
    df = pd.read_csv(filepath_or_buffer='dataframe.csv')
    print(df)
    dataset = FrameDatasetForGnn(
        data_dir='/home/mitarb/plenz/ws/projects/git_shared/SemEval233Framing/data/GraphData/lookup=sbert_1_r2nl=free_kg=CNsmall_pruning=True_k=1_oPCp=False_alledges=False',
        edge_type_combination='pos_neg'
    )
    dataset.init_data(df=df)

    batch = [dataset.__getitem__(index=i) for i in np.random.choice(range(len(dataset)), size=8)]

    # init model
    gnn = GnnEncoder(
        # gnn_layer = 'GCNConv',
        gnn_layer = 'FastRGCNConv',
        num_layers = 6,
        in_channels = 768,
        hidden_channels = 16,
        out_channels = 14,
        activation='ReLU',
        num_relations=2,
    )
    gnn.to('cuda')
    print(gnn)

    # test forward pass
    y = gnn(batch)
    print(f"final: {y = }")
    print(f"final: {y.shape = }")

    print('done')