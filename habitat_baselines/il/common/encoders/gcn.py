import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, EdgeConv, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from habitat import logger

conv_layers = {
    "GCNConv": GCNConv,
    "GATConv": GATConv, 
}

class LocalGCNEncoder(nn.Module):
    def __init__(
        self,
        node_feature_dim,
        hidden_feature_dim,
        out_feature_dim,
        embedding_dim,
        gcn_config,
    ):
        super().__init__()
        self.no_node_cat = gcn_config.no_node_cat
        self.no_bbox_feats = gcn_config.no_bbox_feats
        self.no_gcn = gcn_config.no_gcn

        if self.no_node_cat:
            embedding_dim = 0
        if self.no_bbox_feats:
            node_feature_dim = 0
        
        conv_layer_fn = conv_layers[gcn_config.conv_layer]

        self.conv_1 = conv_layer_fn(node_feature_dim + embedding_dim, hidden_feature_dim)
        self.dropout_1 = nn.Dropout(0.3)

        self.conv_2 = conv_layer_fn(hidden_feature_dim, hidden_feature_dim)
        self.dropout_2 = nn.Dropout(0.3)
        
        self.conv_3 = conv_layer_fn(hidden_feature_dim, hidden_feature_dim)
        self.dropout_3 = nn.Dropout(0.3)

        self.edge_conv_1 = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * hidden_feature_dim, hidden_feature_dim),
                nn.ReLU(),
                nn.Linear(hidden_feature_dim, hidden_feature_dim),
            )
        )

        # self.edge_conv_2 = EdgeConv(
        #     nn.Sequential(
        #         nn.Linear(2 * hidden_feature_dim, hidden_feature_dim),
        #         nn.ReLU(),
        #         nn.Linear(hidden_feature_dim, hidden_feature_dim),
        #     )
        # )

        self.fc_1 = nn.Linear(hidden_feature_dim * 2, out_feature_dim)
        self.act_1 = nn.ReLU()
        self.dropout_4 = nn.Dropout(0.5)
        self.fc_2 = nn.Linear(out_feature_dim, out_feature_dim)
        self.act_2 = nn.ReLU()

        if not self.no_node_cat:
            logger.info("No node category input")
            self.cat_embedding = nn.Embedding(1496, embedding_dim)

        logger.info("GCN hidden dim: {}".format(hidden_feature_dim))

    def forward(self, local_scene_graph, local_scene_graph_index):
        x = local_scene_graph.nodes
        edge_index = local_scene_graph.edge_index
        batch = local_scene_graph_index

        feats = []
        # test detector features
        if self.no_gcn:
            cat_feats = gap(x, batch).float()
            return cat_feats
        
        if not self.no_bbox_feats:
            feats.append(x)

        if not self.no_node_cat:
            node_cat = self.cat_embedding(local_scene_graph.node_categories)
            feats.append(node_cat)

        x = torch.cat(feats, dim=1).float()

        x = F.relu(self.conv_1(x, edge_index))

        x = F.relu(self.conv_2(x, edge_index))
     
        x = F.relu(self.conv_3(x, edge_index))

        x = F.relu(self.edge_conv_1(x, edge_index))

        # x = F.relu(self.edge_conv_2(x, edge_index))

        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
     
        x = self.dropout_4(self.act_1(self.fc_1(x)))
        x = self.act_2(self.fc_2(x))
        return x
        