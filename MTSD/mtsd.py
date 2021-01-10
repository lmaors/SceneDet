import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MTSDone(nn.Module):
    def __init__(self, cfg, mode="place"):
        super(MTSDone, self).__init__()
        self.seq_len = cfg.seq_len
        self.embed_size = cfg.model.embed_size
        self.ncls =  cfg.model.ncls
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
        elif mode == "sub":
            self.bnet = BNet(cfg)
            self.input_dim = (cfg.model.sub_feat_dim + cfg.model.sim_channel)
        else:
            pass
        self.fc1 = nn.Linear(self.input_dim, self.embed_size)
        self.shot_transformer = ShotTransformer(cfg)


    def forward(self, src):
        src = self.bnet(src)
        src = src.view(-1, self.seq_len, src.shape[-1])
        if self.input_dim != self.embed_size:
            src = self.fc1(src)
        out = self.shot_transformer(src,has_mask=False)
        out = out.view(-1, self.ncls)
        return out


class MTSD(nn.Module):
    def __init__(self, cfg):
        super(MTSD, self).__init__()
        self.seq_len = cfg.seq_len
        self.mode = cfg.dataset.mode
        self.ratio = torch.tensor(cfg.model.ratio)
        if 'place' in self.mode:
            self.bnet_place = MTSDone(cfg, "place")
        if 'cast' in self.mode:
            self.bnet_cast = MTSDone(cfg, "cast")
        if 'act' in self.mode:
            self.bnet_act = MTSDone(cfg, "act")
        if 'aud' in self.mode:
            self.bnet_aud = MTSDone(cfg, "aud")
        if 'sub' in self.mode:
            self.bnet_sub = MTSDone(cfg,'sub')

    def forward(self, place_feat, cast_feat, act_feat, aud_feat, sub_feat):
        out = 0
        ratio = F.softmax(self.ratio,dim=0)
        # ratio = self.ratio
        if 'place' in self.mode:
            place_bound = self.bnet_place(place_feat)
            out += ratio[0] * place_bound
        if 'cast' in self.mode:
            cast_bound = self.bnet_cast(cast_feat)
            out += ratio[1] * cast_bound
        if 'act' in self.mode:
            act_bound = self.bnet_act(act_feat)
            out += ratio[2] * act_bound
        if 'aud' in self.mode:
            aud_bound = self.bnet_aud(aud_feat)
            out += ratio[3] * aud_bound
        if 'sub' in self.mode:
            sub_bound = self.bnet_sub(sub_feat)
            out += ratio[4] * sub_bound
        return out

# sin_cos 编码
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# 直接 position embeddings
class PositionalEmbedding(nn.Module):
    def __init__(self,cfg):
        super(PositionalEmbedding, self).__init__()
        self.seq_len = cfg.model.transformer_seq_len
        self.embed_size = cfg.model.embed_size
        self.position_embedding = nn.Embedding(self.seq_len, self.embed_size)
        self.layernorm = nn.LayerNorm(self.embed_size)
        self.dropout = nn.Dropout(cfg.model.transformer_dropout)
    def forward(self, x):
        position_ids = torch.arange(self.seq_len, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand(x.size(0),x.size(1)).cuda()
        position_embeddings = self.position_embedding(position_ids)
        embeddings = x + position_embeddings
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class ShotTransformer(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self,cfg):
        super(ShotTransformer, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.ninp=cfg.model.embed_size
        self.nhead = cfg.model.transformer_attn_heads
        self.dropout=cfg.model.transformer_dropout
        self.nhid=cfg.model.transformer_dim_feedforward  
        self.ncls=2   # labels:1, 0   1:scene boundary  0: not scene boundary
        self.activation = cfg.model.transformer_activation
        self.nlayers = cfg.model.transformer_n_layers
        # self.pos_encoder = PositionalEncoding(self.ninp, self.dropout)
        self.pos_embedding = PositionalEmbedding(cfg)
        encoder_layers = TransformerEncoderLayer(self.ninp, self.nhead, self.nhid, self.dropout, self.activation)
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.nlayers)
        self.decoder = nn.Sequential(
            nn.LayerNorm(self.ninp),
            nn.Linear(self.ninp, 2*self.ninp),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(2*self.ninp, self.ncls)
        )
        # self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=False):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
        src = self.pos_embedding(src) 
        src = src.permute(1,0,2)   # shape(seq_len, batch_size, embed_size)
        out = self.transformer_encoder(src, self.src_mask)
        out = out.permute(1,0,2)     # shape(batch_size, seq_len, embed_size)
        out = self.decoder(out)
        return out


if __name__ == '__main__':
    from mmcv import Config

    cfg = Config.fromfile("./config.py")
    model = MTSD(cfg)
    place_feat = torch.randn(cfg.batch_size, cfg.seq_len, cfg.shot_num, 2048)
    cast_feat = torch.randn(cfg.batch_size, cfg.seq_len, cfg.shot_num, 512)
    act_feat = torch.randn(cfg.batch_size, cfg.seq_len, cfg.shot_num, 512)
    aud_feat = torch.randn(cfg.batch_size, cfg.seq_len, cfg.shot_num, 257, 90)
    sub_feat = torch.randn(cfg.batch_size, cfg.seq_len, cfg.shot_num, 768)
    tgt_feat = torch.randint(0,2,(cfg.batch_size, 10))

    output = model(place_feat, cast_feat, act_feat, aud_feat, sub_feat)
    # output = output.view(-1, 2)
    # target = tgt_feat.view(-1)
    # output = F.softmax(output, dim=1)
    # prob = output[:, 1]
    # loss = nn.CrossEntropyLoss(torch.Tensor(cfg.loss.weight))(output, target)
    print(output[:5])
    print(cfg.batch_size)
    print(output.data.size())


