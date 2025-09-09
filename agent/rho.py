# chatgpt helped me adapt my previous rho for pyg integration 
import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, TransformerConv, global_mean_pool
from torch_geometric.data import HeteroData

"""
- Node types:   'agent', 'scene'
- Edge types:   ('agent','to','scene'), ('scene','to','agent') 
"""
class Rho(nn.Module):
 
    def __init__(
        self,
        in_dim_agent: int,    
        in_dim_scene: int,    
        edge_dim: int,        
        hidden_dim: int = 256,
        out_dim: int = 256,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.0,
        use_agent_agent: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.use_agent_agent = use_agent_agent

        # Type-specific input projections to hidden
        self.in_proj = nn.ModuleDict({
            'agent': nn.Linear(in_dim_agent, hidden_dim),
            'scene': nn.Linear(in_dim_scene, hidden_dim),
        })

        # Build a stack of heterogeneous TransformerConv layers
        # For each layer we create a HeteroConv with one TransformerConv per relation.
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            convs = {}
            # agent -> scene
            convs[('agent','to','scene')] = TransformerConv(
                in_channels=(hidden_dim, hidden_dim),
                out_channels=hidden_dim // heads,
                heads=heads,
                edge_dim=edge_dim,
                dropout=dropout,
                beta=False  
            )
            # scene -> agent
            convs[('scene','to','agent')] = TransformerConv(
                in_channels=(hidden_dim, hidden_dim),
                out_channels=hidden_dim // heads,
                heads=heads,
                edge_dim=edge_dim,
                dropout=dropout,
                beta=False
            )
            # optional agent -> agent scaffold
            if use_agent_agent:
                convs[('agent','to','agent')] = TransformerConv(
                    in_channels=(hidden_dim, hidden_dim),
                    out_channels=hidden_dim // heads,
                    heads=heads,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    beta=False
                )

            self.layers.append(HeteroConv(convs, aggr='sum'))

        # Output projection per type (hidden -> out)
        self.out_proj = nn.ModuleDict({
            'agent': nn.Linear(hidden_dim, out_dim),
            'scene': nn.Linear(hidden_dim, out_dim),
        })

        self.act = nn.SiLU()
        self.norm = nn.ModuleDict({
            'agent': nn.LayerNorm(hidden_dim),
            'scene': nn.LayerNorm(hidden_dim),
        })

    def forward(self, data: HeteroData):
        x_dict = {
            'agent': self.act(self.in_proj['agent'](data['agent'].x)),
            'scene': self.act(self.in_proj['scene'](data['scene'].x)),
        }

        # run hetero conv stack
        for hetero_conv in self.layers:
            # Build one dict per kwarg expected by the underlying convs
            edge_attr_dict = {}
            for rel in hetero_conv.convs.keys():
                # rel is a tuple like ('agent','to','scene')
                edge_attr_dict[rel] = data[rel].edge_attr

            x_dict = hetero_conv(
                x_dict,
                data.edge_index_dict,
                edge_attr_dict=edge_attr_dict
            )

            # per-type norm + activation
            for ntype in x_dict:
                x_dict[ntype] = self.norm[ntype](x_dict[ntype])
                x_dict[ntype] = self.act(x_dict[ntype])

        # project to out_dim
        node_emb = {
            'agent': self.out_proj['agent'](x_dict['agent']),   
            'scene': self.out_proj['scene'](x_dict['scene']),   
        }

        # per-graph pooling
        batch_agent = data['agent'].batch if 'batch' in data['agent'] else torch.zeros(
            node_emb['agent'].size(0), dtype=torch.long, device=node_emb['agent'].device
        )
        batch_scene = data['scene'].batch if 'batch' in data['scene'] else torch.zeros(
            node_emb['scene'].size(0), dtype=torch.long, device=node_emb['scene'].device
        )

        pooled_agent = global_mean_pool(node_emb['agent'], batch_agent)   
        pooled_scene = global_mean_pool(node_emb['scene'], batch_scene)   
        pooled_concat = torch.cat([pooled_agent, pooled_scene], dim=-1)   

        pooled = {
            'agent': pooled_agent,
            'scene': pooled_scene,
            'concat': pooled_concat,
        }
        return node_emb, pooled
