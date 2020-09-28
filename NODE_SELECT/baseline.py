import random, torch, time, numpy as np
import torch.nn.functional as F 
import torch.nn as nn
import math

from torch.nn   import Linear,Dropout
from torch.nn   import Parameter 

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.nn import GATConv,GCNConv,GraphConv,SAGEConv, GatedGraphConv, ChebConv
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros

from torch_scatter import scatter_mean
from torch_geometric.data import Data
torch.cuda.empty_cache()

class SIMPLE_LAYER(torch.nn.Module):
    def __init__(self,feat_in,feat_out): 
        super(SIMPLE_LAYER, self).__init__() 
        self.temp_layer   = Linear(feat_in,feat_out)
    def forward(self,x,edge_index): return self.temp_layer(x)

class BaselineNet(torch.nn.Module):
    def __init__(self,feat_in,feat_mid,feat_out, n_layer=1,architecture=1,heads=4):
        super(BaselineNet, self).__init__()
        self.heads        = heads
        self.n_layer      = n_layer
        self.arch         = architecture

        if   architecture == 1: layer = GCNConv
        elif architecture == 2: layer = SAGEConv
        elif architecture == 3: layer = GATConv
        elif architecture == 7: layer = SIMPLE_LAYER

        if architecture   == 3:
            if n_layer > 1 :
                self.conv_in     = layer(feat_in, feat_mid,heads=self.heads)
                self.B_layers    = nn.ModuleList([layer(self.heads*feat_mid,feat_mid,heads=self.heads) for i in range(n_layer-2)])
                self.conv_out    = layer(self.heads*feat_mid, feat_out,concat=False, heads=1)
            else:
                self.conv_in     = layer(feat_in, feat_out,heads=self.heads,concat=False)
                self.B_layers    = []
        else:
            if n_layer > 1 :
                self.conv_in     = layer(feat_in, feat_mid)
                self.B_layers    = nn.ModuleList([layer(feat_mid,feat_mid) for i in range(n_layer-2)])
                self.conv_out    = layer(feat_mid, feat_out)
            else:
                self.conv_in     = layer(feat_in, feat_out)
                self.B_layers    = []


    def forward(self, data):
        x, edge_index  = data.x, data.edge_index

        # FIRST LAYER
        if self.arch == 3:
            x = F.dropout(x, p=0.6, training=self.training)
            x = F.elu(self.conv_in(x, edge_index))
            x = F.dropout(x, p=0.4, training=self.training)
        else:        
            x = self.conv_in(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)    
        
        # HIDDEN LAYERS
        if len(self.B_layers) > 0 :
            for i in range(len(self.B_layers)):
                if self.arch == 3: 
                    x = self.B_layers[i](x, edge_index) 
                    x = F.elu(x)         
                    x = F.dropout(x,p=0.6, training=self.training)                            
                else:
                    x = self.B_layers[i](x, edge_index)
                    x = F.relu(x)
                    x = F.dropout(x,p=0.4, training=self.training)    

        # OUT-LAYER
        if self.n_layer > 1:                    
            x   = self.conv_out(x, edge_index)

        y    = F.log_softmax(x, dim=-1)
        return y 
