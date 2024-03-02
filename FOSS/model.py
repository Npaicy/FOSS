import torch
import numpy as np
from gymnasium import spaces
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.torch_utils import FLOAT_MIN
from config import Config

config = Config()

class FeatureEmbed(nn.Module):
    def __init__(self, embed_size = 32, tables = 22, types = config.types, columns = config.columns, \
                 ops = 12, pos = 4):
        super(FeatureEmbed, self).__init__()
        self.typeEmbed = nn.Embedding(types, embed_size)
        self.tableEmbed = nn.Embedding(tables, embed_size)

        self.columnEmbed = nn.Embedding(columns, 2 * embed_size)
        self.opEmbed = nn.Embedding(ops, embed_size // 8)
        self.posEmbed = nn.Embedding(pos, embed_size // 8)

        self.linearFilter2 = nn.Linear(2 * embed_size  + embed_size // 8,
                                       2 * embed_size  + embed_size // 8)
        self.linearFilter = nn.Linear(2 * embed_size  + embed_size // 8,
                                      2 * embed_size  + embed_size // 8) #        
        self.linearJoin1 = nn.Linear(2 * config.maxjoins  * embed_size,   3 * embed_size)
        self.linearJoin2 = nn.Linear(3 * embed_size,  3 * embed_size)
        self.linearest = nn.Linear(3, embed_size // 4)
        self.project = nn.Linear(
                embed_size * 7 + 4 * (embed_size // 8),
                embed_size * 7 + 4 * (embed_size // 8))
    def forward(self, feature):
        typeId, join, filtersId, filtersMask, posId,table,db_est = torch.split(
                feature, (1, config.maxjoins, 6, 3, 1, 1,3), dim=-1)
        # print(db_est)
        typeEmb = self.getType(typeId)
        joinEmb = self.getJoin(join)
        filterEmbed = self.getFilter(filtersId, filtersMask)
        dbest = self.linearest(db_est)
        tableEmb = self.getTable(table)
        posEmb = self.getPos(posId)
        final = torch.cat((typeEmb, filterEmbed, joinEmb, tableEmb, posEmb,dbest),dim=1)
        temp = self.project(final)
        final = F.leaky_relu(temp)       
        return final

    def getType(self, typeId):
        typeId = typeId.long()
        emb = self.typeEmbed(typeId).squeeze(1)
        return emb

    def getTable(self, table_sample):
        table = table_sample.long()
        emb = self.tableEmbed(table).squeeze(1)
        return emb
    
    def getJoin(self, joins):
        joins = joins.long()
        joins_embed = self.columnEmbed(joins)
        joins_embed = torch.cat([joins_embed[:, i, :] for i in range(config.maxjoins)], dim=-1)

        concat = F.leaky_relu(self.linearJoin1(joins_embed))
        concat = F.leaky_relu(self.linearJoin2(concat))
        concat = concat.squeeze(1)
        return concat
    

    def getPos(self, posId):
        posId = posId.long()
        emb = self.posEmbed(posId).squeeze(1)
        return emb
    
    def getFilter(self, filtersId, filtersMask):
        ## get Filters, then apply mask
        filterExpand = filtersId.view(-1, 2, 3).transpose(1, 2)
        colsId = filterExpand[:, :, 0].long()
        opsId = filterExpand[:, :, 1].long()
        # vals = filterExpand[:, :, 2].unsqueeze(-1)  # b by 3 by 1

        # b by 3 by embed_dim

        col = self.columnEmbed(colsId)
        op = self.opEmbed(opsId)

        # concat = torch.cat((col, op, vals), dim=-1)
        concat = torch.cat((col, op), dim=-1)
        concat = F.leaky_relu(self.linearFilter(concat))
        concat = F.leaky_relu(self.linearFilter2(concat))

        ## apply mask
        concat[~filtersMask.bool()] = 0.
        ## avg by # of filters
        num_filters = torch.sum(filtersMask, dim=1) + 1e-10
        total = torch.sum(concat, dim=1)
        avg = total / num_filters.view(-1, 1)
        return avg

class PlanNetwork(nn.Module):
    def __init__(self, emb_size = config.emb_size ,ffn_dim = config.ffn_dim, \
                 head_size = config.head_size, dropout = 0.05, \
                 attention_dropout_rate = 0.05, n_layers = config.num_layers):

        super(PlanNetwork, self).__init__()

        self.hidden_dim = config.hidden_dim
        self.head_size = head_size
        self.emb_size = emb_size
        self.height_size = emb_size // 2
        self.height_encoder = nn.Embedding(config.heightsize, self.height_size , padding_idx=0)
        self.input_dropout = nn.Dropout(dropout)
        encoders = [
            EncoderLayer(self.hidden_dim, ffn_dim, dropout, attention_dropout_rate,
                         head_size) for _ in range(n_layers)
        ]
        self.layers = nn.ModuleList(encoders)

        self.final_ln = nn.LayerNorm(self.hidden_dim)

        self.embbed_layer = FeatureEmbed(embed_size = emb_size, tables = config.tablenum, ops = config.opsnum)

    def forward(self, batched_data):
        attn_bias, x = batched_data['attn_bias'], batched_data['x']
        heights = batched_data['heights'].long()
        n_batch, n_node = x.size()[:2]
        tree_attn_bias = attn_bias.clone()
        tree_attn_bias = tree_attn_bias.unsqueeze(1).repeat(1, self.head_size, 1, 1)
        tree_attn_bias = tree_attn_bias[:, :, 1:, 1:]
        x_view = x.contiguous().view(-1, 10 + config.maxjoins + 5)
        node_feature = self.embbed_layer(x_view).view(
            n_batch, -1, self.hidden_dim - self.height_size)
        height_feature = self.height_encoder(heights)
        node_feature = torch.cat([node_feature, height_feature], dim=2)
        output = self.input_dropout(node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, tree_attn_bias)
        output = self.final_ln(output)

        return output[:, 0, :] 


class FeedForwardNetwork(nn.Module):

    def __init__(self, hidden_size, ffn_size):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, hidden_size, attention_dropout_rate, head_size):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size**-0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            attn_bias = attn_bias
            #x = x + attn_bias
            x = x * attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):

    def __init__(self, hidden_size, ffn_size, dropout_rate,
                 attention_dropout_rate, head_size):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size,
                                                 attention_dropout_rate,
                                                 head_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x) 
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x

class CustomModel(TorchModelV2,nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self,obs_space, action_space, num_outputs,
                                          model_config, name)
        nn.Module.__init__(self)
        
        self.embedmodel = PlanNetwork()
        obs_space = spaces.Box(-np.inf,np.inf,dtype = np.float32,shape = (self.embedmodel.hidden_dim + 1,))
        self.model = TorchFC(obs_space, action_space,
                                           num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        action_mask = input_dict["obs"]["action_mask"]
        steps = input_dict["obs"]["steps"]
        
        del input_dict["obs"]["action_mask"]
        represnetation = self.embedmodel(input_dict["obs"])
        represnetation = torch.cat((represnetation,steps),dim=-1)
        action_embed, _ = self.model({
            "obs": represnetation
        })
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        return action_embed + inf_mask, state
    def value_function(self):
        return self.model.value_function()
