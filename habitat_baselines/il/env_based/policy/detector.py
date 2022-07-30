import sys
import time
import torch
import numpy as np

from torch import nn
from torch_geometric.data import Data, Batch

from habitat import Config, logger
from habitat.tasks.nav.object_nav_task import task_cat2hm3dcat40, mapping_mpcat40_to_goal21
from habitat_baselines.il.env_based.policy.rednet import load_rednet

from mmdet.apis import init_detector, inference_detector # needs MMDetection library

from scripts.utils.utils import load_json_dataset


class InstanceDetector(nn.Module):
    r"""A wrapper over object detector network.
    """

    def __init__(self, model_config: Config, device):
        super().__init__()
        self.model_config = model_config
        self.detector = None
        self.device = device

        self.max_nodes = 50
        self.filtered_objects = load_json_dataset("configs/detector/filtered_objects_mmdet.json")

        self.filter = model_config.SPATIAL_ENCODER.filter_nodes

        if model_config.DETECTOR.name == "mask_rcnn":
            # Default to Mask RCNN predictor
            self.detector = init_detector(model_config.DETECTOR.config_path, model_config.DETECTOR.checkpoint_path, device)

        self.eval()

    def create_graph_batch(self, graphs):
        graphs = [Data(nodes=graphs[i][0], edge_index=graphs[i][1], node_categories=graphs[i][2]) for i in range(len(graphs))]
        return graphs

    def forward(self, observations):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
        rgb_obs = observations["rgb"]

        if self.model_config.DETECTOR.name == "mask_rcnn":
            batch_start_time = time.time()
            rgb_obs_list = [rgb_obs[i].cpu().numpy() for i in range(rgb_obs.shape[0])]
            batch_end_time = time.time() - batch_start_time

            x, bt_time, ct_time, inf_time = inference_detector(self.detector, rgb_obs_list)

            batch_start_time = time.time()

            graphs = []
            for i in range(len(x)):
                class_idxs = []
                nodes = []

                if self.filter:
                    for j in self.filtered_objects:
                        if x[i][0][j].shape[0] > 0:
                            class_idxs.extend([j] * x[i][0][j].shape[0])
                            nodes.append(x[i][2][j])
                else:
                    for j in range(len(x[i][0])):
                        if x[i][0][j].shape[0] > 0:
                            class_idxs.extend([j] * x[i][0][j].shape[0])
                            nodes.append(x[i][2][j])

                if len(nodes) == 0:
                    nodes.append(np.zeros((1, 1024)))
                    class_idxs = [0]

                nodes = np.concatenate(nodes, axis=0)
                node_categories = np.array(class_idxs)

                nodes_no_filter = np.concatenate(x[i][2], axis=0)
                #logger.info("nodes_filter:{} , no filter: {}".format(nodes.shape, nodes_no_filter.shape))

                # if nodes.shape[0] == 0:
                #     nodes = np.zeros((1, 1024))
                #     node_categories = np.array([0])

                # building edges
                num_nodes = nodes.shape[0]
                    
                adj = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)
                edges = np.transpose(np.argwhere(adj == 1))

                nodes = torch.tensor(nodes).to(self.device)
                edges = torch.tensor(edges).long().to(self.device)
                node_categories = torch.tensor(node_categories).to(self.device)

                graphs.append((nodes, edges, node_categories,))

            edge_bt_time = time.time() - batch_start_time

            gf_start_time = time.time()
            gf_batch = self.create_graph_batch(graphs)
            gf_end_time = time.time() - gf_start_time
            # logger.info("nodes: {}, node_cats: {}, edges: {}".format(x.shape, x_cat.shape, edges.shape))

        return gf_batch, bt_time, ct_time, inf_time, batch_end_time, 0, edge_bt_time, gf_end_time
