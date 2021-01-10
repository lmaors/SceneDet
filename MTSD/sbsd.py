import torch
import torch.nn as nn
import torch.nn.functional as F
from SBSD.embedding.embedding import BoundaryEmbedding
from SBSD.transformer import TransformerBlock


# from SBST.utils.seg_info_process import reduce_segment_info

class AudNet(nn.Module):
    def __init__(self, cfg):
        super(AudNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3),
                               stride=(2, 1), padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3))

        self.conv2 = nn.Conv2d(64, 192, kernel_size=(
            3, 3), stride=(2, 1), padding=0)
        self.bn2 = nn.BatchNorm2d(192)
        self.relu2 = nn.ReLU(inplace=True)
        # self.pool2 = nn.MaxPool2d(kernel_size=(1,3))

        self.conv3 = nn.Conv2d(192, 384, kernel_size=(
            3, 3), stride=(2, 1), padding=0)
        self.bn3 = nn.BatchNorm2d(384)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(384, 256, kernel_size=(
            3, 3), stride=(2, 2), padding=0)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=(
            3, 3), stride=(2, 2), padding=0)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv6 = nn.Conv2d(256, 512, kernel_size=(3, 2), padding=0)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(512, 512)

    def forward(self, x):  # [bs,1,257,90]
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.pool5(self.relu5(self.bn5(self.conv5(x))))
        x = self.relu6(self.bn6(self.conv6(x)))
        x = x.squeeze()
        out = self.fc(x)
        return out


class SubNet(nn.Module):
    def __init__(self, cfg):
        super(SubNet, self).__init__()
        self.conv1 = nn.Conv2d(768, cfg.model.sim_channel, kernel_size=(1, 1))
        self.bnet = BNet(cfg)
        self.fc1 = nn.Linear(512 + cfg.model.sim_channel, 2)

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        out = self.conv1(x)
        out = out.permute(0, 3, 2, 1)
        out = self.bnet(out)
        # out = x.view(-1, self.seq_len, x.shape[-1])
        out = F.relu(self.fc1(out))
        out = out.view(-1, 2)

        return out


class Cos(nn.Module):
    def __init__(self, cfg):
        super(Cos, self).__init__()
        self.shot_num = cfg.shot_num
        self.channel = cfg.model.sim_channel
        self.conv1 = nn.Conv2d(
            1, self.channel, kernel_size=(self.shot_num // 2, 1))

    def forward(self, x):  # [batch_size, seq_len, shot_num, feat_dim]
        x = x.contiguous().view(-1, 1, x.shape[2], x.shape[3])
        part1, part2 = torch.split(x, [self.shot_num // 2] * 2, dim=2)
        # batch_size*seq_len, 1, [self.shot_num//2], feat_dim
        part1 = self.conv1(part1).squeeze()
        part2 = self.conv1(part2).squeeze()
        x = F.cosine_similarity(part1, part2, dim=2)  # batch_size,channel
        return x


class BNet(nn.Module):
    def __init__(self, cfg):
        super(BNet, self).__init__()
        self.shot_num = cfg.shot_num
        self.channel = cfg.model.sim_channel
        self.conv1 = nn.Conv2d(1, self.channel, kernel_size=(cfg.shot_num, 1))
        self.max3d = nn.MaxPool3d(kernel_size=(self.channel, 1, 1))
        self.cos = Cos(cfg)

    def forward(self, x):  # [batch_size, seq_len, shot_num, feat_dim]
        context = x.contiguous().view(
            x.shape[0] * x.shape[1], 1, -1, x.shape[-1])
        context = self.conv1(context)  # batch_size*seq_len,512,1,feat_dim
        context = self.max3d(context)  # batch_size*seq_len,1,1,feat_dim
        context = context.squeeze()
        sim = self.cos(x)
        bound = torch.cat((context, sim), dim=1)
        return bound


class AudBNet(nn.Module):
    def __init__(self, cfg):
        super(AudBNet, self).__init__()
        self.shot_num = cfg.shot_num
        self.channel = cfg.model.sim_channel
        self.audnet = AudNet(cfg)
        self.conv1 = nn.Conv2d(1, self.channel, kernel_size=(cfg.shot_num, 1))
        self.conv2 = nn.Conv2d(
            1, self.channel, kernel_size=(cfg.shot_num // 2, 1))
        self.max3d = nn.MaxPool3d(kernel_size=(self.channel, 1, 1))

    def forward(self, x):  # [batch_size, seq_len, shot_num, 257, 90]
        context = x.view(
            x.shape[0] * x.shape[1] * x.shape[2], 1, x.shape[-2], x.shape[-1])
        context = self.audnet(context).view(
            x.shape[0] * x.shape[1], 1, self.shot_num, -1)
        part1, part2 = torch.split(context, [self.shot_num // 2] * 2, dim=2)
        part1 = self.conv2(part1).squeeze()
        part2 = self.conv2(part2).squeeze()
        sim = F.cosine_similarity(part1, part2, dim=2)
        bound = sim
        return bound


class ShotTransformer(nn.Module):
    # def __init__(self, seq_len=10, hidden=1024, n_layers=12, attn_heads=8, dropout=0.1):
    def __init__(self, cfg):
        super(ShotTransformer, self).__init__()
        self.hidden = cfg.model.transformer_hidden
        self.n_layers = cfg.model.transformer_n_layers
        self.attn_heads = cfg.model.transformer_attn_heads
        self.dropout = cfg.model.transformer_dropout

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = cfg.model.transformer_hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BoundaryEmbedding(
            seq_len=cfg.model.transformer_seq_len, embed_size=cfg.model.transformer_hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden, self.attn_heads, self.hidden * 4, self.dropout) for _ in range(self.n_layers)])

    def forward(self, x):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        y = torch.sum(x, 2)  # 压缩维度 没有实质作用
        # mask = (y > -10000).unsqueeze(1).repeat(1, y.size(1), 1).unsqueeze(1)
        mask = None

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, y)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x


class SBSDone(nn.Module):
    def __init__(self, cfg, mode="place"):
        super(SBSDone, self).__init__()
        self.seq_len = cfg.seq_len
        self.num_layers = 1
        self.n_layers = 4
        # self.lstm_hidden_size = cfg.model.lstm_hidden_size
        self.hidden_size = cfg.model.transformer_hidden
        if mode == "place":
            self.input_dim = (cfg.model.place_feat_dim + cfg.model.sim_channel)
            self.bnet = BNet(cfg)
        elif mode == "cast":
            self.bnet = BNet(cfg)
            self.input_dim = (cfg.model.cast_feat_dim + cfg.model.sim_channel)
        elif mode == "act":
            self.bnet = BNet(cfg)
            self.input_dim = (cfg.model.act_feat_dim + cfg.model.sim_channel)
        elif mode == "aud":
            self.bnet = AudBNet(cfg)
            self.input_dim = cfg.model.aud_feat_dim
        # elif mode == "sub":
        #     self.bnet = BNet(cfg)
        #     self.input_dim = cfg.model.sub_feat_dim

        else:
            pass
        self.fc1 = nn.Linear(self.input_dim, self.hidden_size)
        self.shot_transformer = ShotTransformer(cfg)
        self.fc2 = nn.Linear(self.hidden_size, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.bnet(x)
        x = x.view(-1, self.seq_len, x.shape[-1])
        # torch.Size([128, seq_len, 3*channel])
        # self.lstm.flatten_parameters()
        # out, (_, _) = self.lstm(x, None)
        out = self.shot_transformer(self.fc1(x))
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = out.view(-1, 2)
        return out


class SBSD(nn.Module):
    def __init__(self, cfg):
        super(SBSD, self).__init__()
        self.seq_len = cfg.seq_len
        self.mode = cfg.dataset.mode
        # self.num_layers = 1
        # self.lstm_hidden_size = cfg.model.lstm_hidden_size
        self.ratio = cfg.model.ratio
        if 'place' in self.mode:
            self.bnet_place = SBSDone(cfg, "place")
        if 'cast' in self.mode:
            self.bnet_cast = SBSDone(cfg, "cast")
        if 'act' in self.mode:
            self.bnet_act = SBSDone(cfg, "act")
        if 'aud' in self.mode:
            self.bnet_aud = SBSDone(cfg, "aud")
        if 'sub' in self.mode:
            self.bnet_sub = SubNet(cfg)

    def forward(self, place_feat, cast_feat, act_feat, aud_feat, sub_feat):
        out = 0

        if 'place' in self.mode:
            place_bound = self.bnet_place(place_feat)
            out += self.ratio[0] * place_bound
        if 'cast' in self.mode:
            cast_bound = self.bnet_cast(cast_feat)
            out += self.ratio[1] * cast_bound
        if 'act' in self.mode:
            act_bound = self.bnet_act(act_feat)
            out += self.ratio[2] * act_bound
        if 'aud' in self.mode:
            aud_bound = self.bnet_aud(aud_feat)
            out += self.ratio[3] * aud_bound
        if 'sub' in self.mode:
            sub_bound = self.bnet_sub(sub_feat)
            out += self.ratio[4] * sub_bound
        return out





if __name__ == '__main__':
    from mmcv import Config

    cfg = Config.fromfile("./config.py")
    model = SBSD(cfg)
    place_feat = torch.randn(cfg.batch_size, cfg.seq_len, cfg.shot_num, 2048)
    cast_feat = torch.randn(cfg.batch_size, cfg.seq_len, cfg.shot_num, 512)
    act_feat = torch.randn(cfg.batch_size, cfg.seq_len, cfg.shot_num, 512)
    aud_feat = torch.randn(cfg.batch_size, cfg.seq_len, cfg.shot_num, 257, 90)
    sub_feat = torch.randn(cfg.batch_size, cfg.seq_len, cfg.shot_num, 768)
    # scene_label = torch.randint(0,2,(cfg.batch_size,cfg.seq_len))
    # segment_info = reduce_segment_info(scene_label)
    output = model(place_feat, cast_feat, act_feat, aud_feat, sub_feat)
    print(output[:5])
    # print(cfg.batch_size)
    # print(output.data.size())
