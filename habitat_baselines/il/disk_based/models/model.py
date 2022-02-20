import math
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(
    input_dim: int,
    hidden_dims: Iterable[int],
    output_dim: int,
    use_batchnorm: bool = False,
    dropout: float = 0,
    add_sigmoid: bool = True,
):
    layers = []
    D = input_dim
    if dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(input_dim))
    for dim in hidden_dims:
        layers.append(nn.Linear(D, dim))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(dim))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.ReLU(inplace=True))
        D = dim
    layers.append(nn.Linear(D, output_dim))

    if add_sigmoid:
        layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)


class MultitaskCNN(nn.Module):
    def __init__(
        self,
        num_actions: int = 10,
        only_encoder: bool = False,
        pretrained: bool = True,
        checkpoint_path: str = "data/eqa/eqa_cnn_pretrain/checkpoints/epoch_5.ckpt",
        freeze_encoder: bool = False,
    ):
        super(MultitaskCNN, self).__init__()

        self.num_actions = num_actions
        self.only_encoder = only_encoder

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(4, 8, 5),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(8, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(16, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(32, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(32, 512, 5),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )

        self.linear = nn.Linear(in_features=512 * 32 * 22, out_features=self.num_actions)

        if self.only_encoder:
            if pretrained:
                print("Loading CNN weights from %s" % checkpoint_path)
                checkpoint = torch.load(
                    checkpoint_path, map_location={"cuda:0": "cpu"}
                )
                self.load_state_dict(checkpoint)

                if freeze_encoder:
                    for param in self.parameters():
                        param.requires_grad = False
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = (
                        m.kernel_size[0]
                        * m.kernel_size[1]
                        * (m.out_channels + m.in_channels)
                    )
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, x):

        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)

        if self.only_encoder:
            return conv4.view(-1, 32 * 26 * 36)

        features = self.classifier(conv4)
        features = features.view(-1, 512 * 32 * 22)
        logits = self.linear(features)

        return logits


class InstructionLstmEncoder(nn.Module):
    def __init__(
        self,
        token_to_idx: Dict,
        wordvec_dim: int = 64,
        rnn_dim: int = 64,
        rnn_num_layers: int = 2,
        rnn_dropout: int = 0,
    ):
        super(InstructionLstmEncoder, self).__init__()

        self.token_to_idx = token_to_idx
        self.NULL = token_to_idx["<pad>"]
        self.START = token_to_idx["<s>"]
        self.END = token_to_idx["</s>"]

        self.embed = nn.Embedding(len(token_to_idx), wordvec_dim)
        self.rnn = nn.LSTM(
            wordvec_dim,
            rnn_dim,
            rnn_num_layers,
            dropout=rnn_dropout,
            batch_first=True,
        )

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        N, T = x.size()
        idx = torch.LongTensor(N).fill_(T - 1)

        # Find the last non-null element in each sequence
        idx = (x != self.NULL).long().sum(-1) - 1
        idx = idx.type_as(x.data).long()
        idx.requires_grad = False

        hs, _ = self.rnn(self.embed(x.long()))

        idx = idx.view(N, 1, 1).expand(N, 1, hs.size(2))
        H = hs.size(2)
        return hs.gather(1, idx).view(N, H)


class RearrangementLstmCnnAttentionModel(nn.Module):
    def __init__(
        self,
        instruction_vocab: Dict,
        rearrangement_pretrain_ckpt_path: str,
        freeze_encoder: bool = False,
        image_feat_dim: int = 64,
        instruction_wordvec_dim: int = 64,
        instruction_hidden_dim: int = 64,
        instruction_num_layers: int = 2,
        instruction_dropout: float = 0.5,
        fc_use_batchnorm: bool = False,
        fc_dropout: float = 0.5,
        fc_dims: Iterable = (64,),
        num_actions: int = 10
    ):
        super(RearrangementLstmCnnAttentionModel, self).__init__()

        cnn_kwargs = {
            "num_actions": num_actions,
            "only_encoder": True,
            "pretrained": False,
            "checkpoint_path": rearrangement_pretrain_ckpt_path,
            "freeze_encoder": freeze_encoder,
        }
        self.cnn = MultitaskCNN(**cnn_kwargs)
        self.cnn_fc_layer = nn.Sequential(
            nn.Linear(32 * 26 * 36, 64), nn.ReLU(), nn.Dropout(p=0.5)
        )

        q_rnn_kwargs = {
            "token_to_idx": instruction_vocab,
            "wordvec_dim": instruction_wordvec_dim,
            "rnn_dim": instruction_hidden_dim,
            "rnn_num_layers": instruction_num_layers,
            "rnn_dropout": instruction_dropout,
        }
        self.q_rnn = InstructionLstmEncoder(**q_rnn_kwargs)

        self.img_tr = nn.Sequential(nn.Linear(64, 64), nn.Dropout(p=0.5))

        self.instruction_tr = nn.Sequential(nn.Linear(64, 64), nn.Dropout(p=0.5))

        classifier_kwargs = {
            "input_dim": 64,
            "hidden_dims": fc_dims,
            "output_dim": num_actions,
            "use_batchnorm": True,
            "dropout": fc_dropout,
            "add_sigmoid": False,
        }
        self.classifier = build_mlp(**classifier_kwargs)

        self.att = nn.Sequential(
            nn.Tanh(), nn.Dropout(p=0.5), nn.Linear(128, 1)
        )

    def forward(
        self, images: torch.Tensor, instruction: torch.Tensor
    ) -> Tuple[torch.tensor]:

        images = torch.unsqueeze(images, 1)
        N, T, _, _, _ = images.size()

        # bs x 5 x 3 x 256 x 256
        img_feats = self.cnn(
            images.contiguous().view(
                -1, images.size(2), images.size(3), images.size(4)
            )
        )

        img_feats = self.cnn_fc_layer(img_feats)

        img_feats_tr = self.img_tr(img_feats)
        ques_feats = self.q_rnn(instruction)

        ques_feats_repl = ques_feats.view(N, 1, -1).repeat(1, T, 1)
        ques_feats_repl = ques_feats_repl.view(N * T, -1)

        ques_feats_tr = self.instruction_tr(ques_feats_repl)

        ques_img_feats = torch.cat([ques_feats_tr, img_feats_tr], 1)

        att_feats = self.att(ques_img_feats)
        att_probs = F.softmax(att_feats.view(N, T), dim=1)
        att_probs2 = att_probs.view(N, T, 1).repeat(1, 1, 64)

        att_img_feats = torch.mul(att_probs2, img_feats.view(N, T, 64))
        att_img_feats = torch.sum(att_img_feats, dim=1)

        mul_feats = torch.mul(ques_feats, att_img_feats)

        scores = self.classifier(mul_feats)

        return scores, att_probs