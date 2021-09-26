import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from easydict import EasyDict as edict
from torch.autograd import Variable
from torch.nn.utils import spectral_norm
from torchvision.models.video.resnet import r2plus1d_18

from .miscc.config import cfg
from .recurrent import BertEncoderWithMemory, BertEmbeddings, NonRecurTransformer
from .layers import DynamicFilterLayer1D as DynamicFilterLayer
from .GLAttention import GLAttentionGeneral as ATT_NET
from .cross_attention import LxmertCrossAttentionLayer as CrossAttn
from torchvision import models


# remind me of what the configs are
base_config = edict(
    hidden_size=768,
    vocab_size=None,  # get from word2idx
    video_feature_size=2048,
    max_position_embeddings=None,  # get from max_seq_len
    max_v_len=100,  # max length of the videos
    max_t_len=30,  # max length of the text
    n_memory_cells=10,  # memory size will be (n_memory_cells, D)
    type_vocab_size=2,
    layer_norm_eps=1e-12,  # bert layernorm
    hidden_dropout_prob=0.1,  # applies everywhere except attention
    num_hidden_layers=2,  # number of transformer layers
    attention_probs_dropout_prob=0.1,  # applies only to self attention
    intermediate_size=768,  # after each self attention
    num_attention_heads=12,
    memory_dropout_prob=0.1
)

if torch.cuda.is_available():
    T = torch.cuda
    cuda_is_available = True
else:
    T = torch
    cuda_is_available = False

def conv1x1(in_planes, out_planes, bias=False): # Moda: only for MART, used in "NEXT_STAGE_G" class
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)

def conv3x3(in_planes, out_planes, stride=1, use_spectral_norm=False): # Moda: use_spectral_norm is added as per CP-CSV (NoCascade). Mostly not needed bcs other classes in this module do not use it so far!
    "3x3 convolution with padding"
    if use_spectral_norm:
        return spectral_norm(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False))
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

# Upscale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True))
    return block

def downBlock(in_planes, out_planes): # Moda: Only for CP-CSV (Cascade) model
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True))
    return block

class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])

class ResBlock(nn.Module): # Moda: only for MART
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(True),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out

class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = cfg.TEXT.DIMENSION * cfg.VIDEO_LEN
        self.c_dim = cfg.GAN.CONDITION_DIM
        self.fc = nn.Linear(self.t_dim, self.c_dim * 2, bias=True)
        self.relu = nn.ReLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if cfg.CUDA and cuda_is_available: #Moda-fix: add cuda condition
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar

class MartPooler(nn.Module): # Moda: only for MART
    def __init__(self, input_size, output_size, mode='attention'):
        super(MartPooler, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_size, output_size),
                                nn.BatchNorm1d(output_size),
                                nn.ReLU(True))
        self.mode = 'attention'
        if mode == 'attention':
            self.context_vector = nn.Parameter(torch.rand(1, 1, input_size))
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_states, mask):
        if self.mode == 'attention':
            attention_score = torch.bmm(self.context_vector.repeat(input_states.shape[0], 1, 1),
                                        input_states.transpose(1, 2).contiguous())
            attention_weights = self.softmax(attention_score.squeeze())*mask.float()
            pooled_output = torch.sum(attention_weights[:, :, None]*input_states, dim=1)
        else:
            seq_lens_sqrt = torch.sqrt(torch.sum(mask, dim=-1).float()).unsqueeze(-1).repeat(1, input_states.shape[-1])
            pooled_output = torch.sum(input_states*mask.unsqueeze(-1).repeat(1, 1, input_states.shape[-1]).float(), dim=-2)/seq_lens_sqrt
        return self.fc(pooled_output)

class NEXT_STAGE_G(nn.Module): # Moda: only for MART, used in "StoryMartGAN" class
    def __init__(self, ngf, nef, ncf, attn_type="cross"):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = nef
        self.cf_dim = ncf
        self.attn_type = attn_type
        # print(ngf, nef, ncf)  (32, 256, 100)
        # (32, 256, 100)
        self.num_residual = cfg.GAN.R_NUM
        self.define_module()
        self.conv = conv1x1(ngf * 3, ngf * 2)

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.GAN.R_NUM): # 2
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        if self.attn_type == "stream":
            self.att = ATT_NET(ngf, self.ef_dim)
        elif self.attn_type == "cross":
            self.att = CrossAttn(ngf, self.ef_dim)
        else:
            raise ValueError
        self.residual = self._make_layer(ResBlock, ngf * 2)
        self.upsample = upBlock(ngf * 2, ngf)

    def forward(self, h_code, c_code, word_embs, mask):
        """
            h_code1(query):  batch x idf x ih x iw (queryL=ihxiw)
            word_embs(context): batch x cdf x sourceL (sourceL=seq_len)
            c_code1: batch x idf x queryL
            att1: batch x sourceL x queryL
        """
        # print(word_embs.shape)
        if self.attn_type == "stream":
            # print('========')
            # ((16, 32, 64, 64), (16, 100), (16, 256, 18), (16, 18))
            # print(h_code.size(), c_code.size(), word_embs.size(), mask.size())
            # here, a new c_code is generated by self.att() method.
            # weightedContext, weightedSentence, word_attn, sent_vs_att
            # print(h_code.shape, c_code.shape, word_embs.shape, mask.shape)
            # c_code, weightedSentence, att, sent_att = self.att(h_code, c_code, word_embs, mask)
            c_code, weightedSentence, _, _ = self.att(h_code, c_code, word_embs, mask)
            # print(c_code.shape, weightedSentence.shape)
            # Then, image feature are concated with a new c_code, they become h_c_code,
            # so, here I can make some change, to concate more items together.
            # which means I need to get more output from line 369, self.att()
            # also, I need to feed more information to calculate the function, and let's see what the new idea will return.
            h_c_code = torch.cat((h_code, c_code), 1)
            # print(h_c_code.shape)
            # print('h_c_code.size:', h_c_code.size())  # ('h_c_code.size:', (16, 64, 64, 64))
            h_c_sent_code = torch.cat((h_c_code, weightedSentence), 1)
            # print(h_c_sent_code.shape)
            # print('h_c_sent_code.size:', h_c_sent_code.size())
            # ('h_c_code.size:', (16, 64, 64, 64))
            # ('h_c_sent_code.size:', (16, 96, 64, 64))
            h_c_sent_code = self.conv(h_c_sent_code)
        elif self.attn_type == "cross":
            c_code, _ = self.att(h_code, word_embs, mask)
            h_c_sent_code = torch.cat((h_code, c_code), 1)
        else:
            raise ValueError

        # print(h_c_sent_code.shape)
        out_code = self.residual(h_c_sent_code)
        # print(out_code.shape)
        # print('out_code:', out_code.size())
        # state size ngf/2 x 2in_size x 2in_size
        out_code = self.upsample(out_code)
        return out_code, None

class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf, nef, bcondition=True):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition
        self.pooler = torch.nn.MaxPool2d(2, stride=2)
        if bcondition:
            self.outlogits = nn.Sequential(
                conv3x3(ndf * 8 + nef, ndf * 8), # Moda: in CP-CSV (NoCascade) use_spectral_norm is True-> conv3x3(ndf * 8 + nef, ndf * 8, use_spectral_norm=True),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())
        else:
            self.outlogits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())

    def forward(self, h_code, c_code=None):
        if h_code.shape[-1] == 8: # Moda: this branch is only for MART
            # print('Before pooling', h_code.shape)
            h_code = self.pooler(h_code)
            # print('After pooling', h_code.shape)
        # conditioning output
        if self.bcondition and c_code is not None:
            # print(h_code.shape, c_code.shape)
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((h_code, c_code), 1)
        else:
            h_c_code = h_code

        output = self.outlogits(h_c_code)
        return output.view(-1)

class VideoEncoder(nn.Module): # Moda: from CP-CSV (NoCascade), used in "STAGE1_D_STY_V2" class
    def __init__(self):
        super(VideoEncoder, self).__init__()
        video_resnet = r2plus1d_18(pretrained=False, progress=True)
        padding= 1
        block = [
            R2Plus1dStem(),
            spectral_norm(nn.Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, padding, padding)
                ,bias=False)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv3d(128, 128, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(padding, 0, 0),
                            bias=False)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv3d(128, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, padding, padding),
                bias=False)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv3d(128, 256, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(padding, 0, 0),
                bias=False)),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv3d(256, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, padding, padding),
                bias=False)),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv3d(256, 512, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(padding, 0, 0),
                bias=False)),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv3d(512, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, padding, padding),
                bias=False)),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv3d(512, 512, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(padding, 0, 0),
                bias=False)),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2),
        ]
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.story_encoder = nn.Sequential(*block)
        self.detector = nn.Sequential(
            spectral_norm(nn.Linear(512, 128)),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            spectral_norm(nn.Linear(128, 1)),
        )

    def forward(self, story):
        '''
            story: B x T X C X W X H
            B: batch size, N : number of story, T story length
            C: image channel, WxH width and height
        '''
        B = story.shape[0]
        latents = self.story_encoder(story)
        latents = self.pool(latents)
        latents = latents.view(B, -1)
        return self.detector(latents)

class BasicBlock(nn.Module): # Moda: Extra class from CP-CSV (NoCascade), not used in this module (redundant)! But it is available there in story_fid_model.py

    __constants__ = ['downsample']
    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes),
            nn.BatchNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class R2Plus1dStem(nn.Sequential): # Moda: from CP-CSV (NoCascade), used in "VideoEncoder" class
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution
    """
    def __init__(self):
        super(R2Plus1dStem, self).__init__(
            spectral_norm(nn.Conv3d(3, 45, kernel_size=(1, 7, 7),
                      stride=(1, 2, 2), padding=(0, 3, 3),
                      bias=False)),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv3d(45, 64, kernel_size=(1, 1, 1),
                      stride=(1, 1, 1), padding=(1, 0, 0),
                      bias=False)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))

# ############# Networks for stageI GAN #############
def CreateModel(Mart=False, Cascade=False, cfg=None, video_len=None): # Mode: new function to create the target model
    if not Mart:
        if not Cascade:
            print("No MART No Cascade Model is created")
            return NoMartNoCascade(cfg, video_len)
        else:
            print("No MART Cascade Model is created")
            return NoMartCascade(cfg, video_len)
    else:
        if not Cascade:
            print("MART No Cascade Model is created")
            return MartNoCascade(cfg, video_len)
        else:
            print("MART Cascade Model is created")
            return MartCascade(cfg, video_len)

class StoryGAN(nn.Module):
    def __init__(self, cfg, video_len):
        super(StoryGAN, self).__init__()
        self.cfg = cfg
        self.gf_dim = cfg.GAN.GF_DIM * 8 # 128*8=1024
        self.motion_dim = cfg.TEXT.DIMENSION + cfg.LABEL_NUM # (356+9=365)
        self.content_dim = cfg.GAN.CONDITION_DIM  # encoded text dim (124)
        self.noise_dim = cfg.GAN.Z_DIM  # noise (100)
        self.recurrent = nn.GRUCell(self.noise_dim + self.motion_dim, self.motion_dim) # (465,365)

        if cfg.USE_TRANSFORMER:
            # Moda: this branch is special for DuCo repo, the else branch matchs CP-CSV (NoCascade) repo
            self.moco_fc = nn.Sequential(nn.Linear(self.motion_dim, self.cfg.MART.hidden_size), nn.ReLU())
            self.mocornn = NonRecurTransformer(cfg.MART, self.content_dim)
            # self.recurrent_fc = nn.Sequential(nn.Linear(self.motion_dim+self.noise_dim, self.cfg.MART.hidden_size), nn.ReLU())
            # self.recurrent = NonRecurTransformer(cfg.MART, self.motion_dim)
        else:
            self.mocornn = nn.GRUCell(self.motion_dim, self.content_dim) # (365,124)
            # self.recurrent = nn.GRUCell(self.noise_dim + self.motion_dim, self.motion_dim)
        
        # change configuration for MART models
        ##########################
        self.moconn = BertEncoderWithMemory(self.cfg.MART)
        self.pooler = MartPooler(self.cfg.MART.hidden_size, self.content_dim)
        self.embeddings = BertEmbeddings(self.cfg.MART, add_postion_embeddings=True)
        if cfg.MART.pretrained_embeddings != '':
            self.embeddings.set_pretrained_embedding(torch.from_numpy(torch.load(cfg.MART.pretrained_embeddings)).float(), cfg.MART.freeze_embeddings)

        self.video_len = video_len
        self.n_channels = 3
        self.filter_num = 3
        self.filter_size = 21
        self.image_size = 124
        self.out_num = 1

        # for segment image v1
        self.use_segment = cfg.SEGMENT_LEARNING
        if self.use_segment:
            # Moda: Adding Seg part
            self.batch_size = cfg.TRAIN.IM_BATCH_SIZE # Moda: redundant, only available in CP-CSV
            self.gf_dim_seg = cfg.GAN.GF_SEG_DIM #512 #48
            # for segment image v1
            self.use_segment = cfg.SEGMENT_LEARNING

            self.segment_size = 4*2*2*2*2 # inital size is 4, upsample 4 times = 64
            self.segment_flat_size = 3*self.segment_size**2 # 12288
        # v2
        # self.aux_size = 5 # Moda: redundant, only available in CP-CSV
        # self.fix_input = 0.1*torch.tensor(range(self.aux_size)).float().cuda() # Moda: redundant, only available in CP-CSV

        self.define_module()

    def define_module(self): # Moda: abstract method
        pass

    def sample_videos(self): # Moda: abstract method
        pass

    def sample_images(self): # Moda: abstract method
        pass

    def get_iteration_input(self, motion_input):
        num_samples = motion_input.shape[0]
        noise = T.FloatTensor(num_samples, self.noise_dim).normal_(0, 1)
        return torch.cat((noise, motion_input), dim=1)

    def get_gru_initial_state(self, num_samples):
        return Variable(T.FloatTensor(num_samples, self.motion_dim).normal_(0, 1))

    def sample_z_motion(self, motion_input, video_len=None): # Moda: video_len is added, now it matchs MART and CP-CSV  (NoCascade)
        # Moda: the original video_len is commented, the new one matchs the MART and CP-CSV  (NoCascade)
        video_len = video_len if video_len is not None else self.video_len
        # video_len = 1 if len(motion_input.shape) == 2 else self.video_len

        num_samples = motion_input.shape[0]

        # if len(motion_input.shape) == 2:
        #     motion_input = motion_input.unsqueeze(1)
        #     filler_input = torch.rand(
        #         (motion_input.shape[0], self.cfg.MART.max_t_len - video_len, motion_input.shape[-1]))
        #     if self.cfg.CUDA:
        #         filler_input = filler_input.cuda()
        #     motion_input = torch.cat((motion_input, filler_input), dim=1)
        #     mask = torch.cat((torch.ones((motion_input.shape[0], video_len)),
        #                       torch.zeros((motion_input.shape[0], self.cfg.MART.max_t_len - video_len))), dim=-1)
        #     # print("Mask shape: ", mask.shape)
        #     # print("max_t_len: ", self.cfg.MART.max_t_len)
        #     # print("Motion Content Shape: ", motion_input.shape)
        # else:
        #     mask = torch.ones((motion_input.shape[0], video_len))
        # if self.cfg.CUDA:
        #     mask = mask.cuda()

        # if self.cfg.USE_TRANSFORMER:
        #     e = torch.cat([self.get_iteration_input(motion_input[:, i, :]).unsqueeze(1) for i in range(self.video_len)], dim=1)
        #     z_motion = self.recurrent(self.recurrent_fc(e), mask)
        #     if video_len == 1:
        #         z_motion = z_motion[:, 0, :]
        #     else:
        #         z_motion = z_motion.view(-1, self.motion_dim)
        # else:

        h_t = [self.m_net(self.get_gru_initial_state(num_samples))]
        for frame_num in range(video_len):
            if len(motion_input.shape) == 2:
                e_t = self.get_iteration_input(motion_input)
            else:
                e_t = self.get_iteration_input(motion_input[:, frame_num, :])
            h_t.append(self.recurrent(e_t, h_t[-1]))
        z_m_t = [h_k.view(-1, 1, self.motion_dim) for h_k in h_t]
        z_motion = torch.cat(z_m_t[1:], dim=1).view(-1, self.motion_dim)
        return z_motion

    def motion_content_rnn(self, motion_input, content_input, dummy_param_1=None, dummy_param_2=None): # Moda: works with NoMART models (match no of param and return to MART version)
        video_len = 1 if len(motion_input.shape) == 2 else self.video_len
        if len(motion_input.shape) == 2:
            motion_input = motion_input.unsqueeze(1)
            filler_input = torch.rand(
                (motion_input.shape[0], self.cfg.MART.max_t_len - video_len, motion_input.shape[-1]))
            if self.cfg.CUDA and cuda_is_available: # Moda-fix: add cuda condition
                filler_input = filler_input.cuda()
            motion_input = torch.cat((motion_input, filler_input), dim=1)
            mask = torch.cat((torch.ones((motion_input.shape[0], video_len)),
                              torch.zeros((motion_input.shape[0], self.cfg.MART.max_t_len - video_len))), dim=-1)
        else:
            mask = torch.ones((motion_input.shape[0], video_len))

        if self.cfg.CUDA and cuda_is_available: # Moda-fix: add cuda condition
            mask = mask.cuda()

        if self.cfg.USE_TRANSFORMER:
            mocornn_co = self.mocornn(self.moco_fc(motion_input), mask).view(-1, self.content_dim)
        else:
            # Moda: this branch is CP-CSV (NoCascade) part
            h_t = [self.c_net(content_input)]
            for frame_num in range(video_len):
                h_t.append(self.mocornn(motion_input[:, frame_num, :], h_t[-1]))
            c_m_t = [h_k.view(-1, 1, self.content_dim) for h_k in h_t]
            mocornn_co = torch.cat(c_m_t[1:], dim=1).view(-1, self.content_dim)

        return mocornn_co, None, None

    def mart_forward_step(self, prev_ms, input_ids, input_masks): # Moda: for MART models
        """single step forward in the recursive structure"""
        embeddings = self.embeddings(input_ids)  # (N, L, D)
        prev_ms, encoded_layer_outputs = self.moconn(prev_ms, embeddings, input_masks, output_all_encoded_layers=False)  # both outputs are list
        return prev_ms, encoded_layer_outputs

class NoMartNoCascade(StoryGAN):
    def define_module(self):
        ninput = self.motion_dim + self.content_dim + self.image_size # (365+124+124=613)
        ngf = self.gf_dim # 128*8=1024

        self.ca_net = CA_NET()
        # -> ngf x 4 x 4

        self.filter_net = nn.Sequential(
            nn.Linear(self.content_dim, self.filter_size * self.filter_num * self.out_num),
            nn.BatchNorm1d(self.filter_size * self.filter_num * self.out_num))

        self.image_net = nn.Sequential(
            nn.Linear(self.motion_dim, self.image_size * self.filter_num),
            nn.BatchNorm1d(self.image_size * self.filter_num),
            nn.Tanh())

        # For generate final image
        self.fc = nn.Sequential(
            nn.Linear(ninput, ngf * 4 * 4, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4),
            nn.ReLU(True))
        # ngf x 4 x 4 -> ngf/2 x 8 x 8
        self.upsample1 = upBlock(ngf, ngf // 2)
        # -> ngf/4 x 16 x 16
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        # -> ngf/8 x 32 x 32
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        # -> ngf/16 x 64 x 64
        self.upsample4 = upBlock(ngf // 8, ngf // 16)
        # -> 3 x 64 x 64
        self.img = nn.Sequential(
            conv3x3(ngf // 16, 3),
            nn.Tanh())

        self.m_net = nn.Sequential(
            nn.Linear(self.motion_dim, self.motion_dim),
            nn.BatchNorm1d(self.motion_dim))

        self.c_net = nn.Sequential(
            nn.Linear(self.content_dim, self.content_dim),
            nn.BatchNorm1d(self.content_dim))

        self.dfn_layer = DynamicFilterLayer(self.filter_size,
                                            pad=self.filter_size // 2)
        # Moda: Adding Seg part
        if self.use_segment:
            ngf_seg = self.gf_dim_seg

            self.seg_c = conv3x3(ngf_seg, ngf)
            self.seg_c1 = conv3x3(ngf_seg//2, ngf//2)
            # self.seg_c2 = conv3x3(ngf_seg//4, ngf//4)
            # self.seg_c3 = conv3x3(ngf_seg//8, ngf//8)
            # self.seg_c4 = conv3x3(ngf_seg//16, ngf//16)

            # For generate seg and img v4 and v5 and v6
            self.fc_seg = nn.Sequential(
                nn.Linear(ninput, ngf_seg * 4 * 4, bias=False),
                nn.BatchNorm1d(ngf_seg * 4 * 4),
                nn.ReLU(True))
            # ngf x 4 x 4 -> ngf/2 x 8 x 8
            self.upsample1_seg = upBlock(ngf_seg, ngf_seg // 2)
            # -> ngf/4 x 16 x 16
            self.upsample2_seg = upBlock(ngf_seg // 2, ngf_seg // 4)
            # -> ngf/8 x 32 x 32
            self.upsample3_seg = upBlock(ngf_seg // 4, ngf_seg // 8)
            # -> ngf/16 x 64 x 64
            self.upsample4_seg = upBlock(ngf_seg // 8, ngf_seg // 16)
            # -> 3 x 64 x 64
            self.img_seg = nn.Sequential(
                conv3x3(ngf_seg // 16, 1),
                nn.Tanh())

    def sample_videos(self, motion_input, content_input, seg=False): # Moda: Seg flag is added, also extra return
        ###
        # motion_input:  batch_size, video_len, 365
        # content_input: batch_size, video_len, 356
        ###
        content_input = content_input.view(-1, cfg.VIDEO_LEN * content_input.shape[2])
        # Moda: the next commented part is from CP-CSV  (NoCascade) (might be needed). Squeezing content_input based on a condition
        # if content_input.shape[0] > 1:
        #     content_input = torch.squeeze(content_input)
        r_code, r_mu, r_logvar = self.ca_net(torch.squeeze(content_input)) ## h0
        c_mu = r_mu.repeat(self.video_len, 1).view(-1, r_mu.shape[1])

        crnn_code, _, _ = self.motion_content_rnn(motion_input, r_code) ## i_t = GRU(s_t) # Moda: add dummy return

        temp = motion_input.view(-1, motion_input.shape[2])
        m_code, m_mu, m_logvar = temp, temp, temp  # self.ca_net(temp)
        m_code = m_code.view(motion_input.shape[0], self.video_len, self.motion_dim)
        zm_code = self.sample_z_motion(m_code) ## *

        # one
        zmc_code = torch.cat((zm_code, c_mu), dim=1)
        # two
        m_image = self.image_net(m_code.view(-1, m_code.shape[2])) ## linearly transform motion(365) to image(372)
        m_image = m_image.view(-1, self.filter_num, self.image_size)
        c_filter = self.filter_net(crnn_code) ## Filter(i_t)
        c_filter = c_filter.view(-1, self.out_num, self.filter_num, self.filter_size)
        mc_image = self.dfn_layer([m_image, c_filter]) ## *

        # Moda: next commented code is DuCo code, it is replaced with CP-CSV  (NoCascade) c_code
        # bcs it contains it in the else branch. But becarful, this function has extra return now
        # zmc_all = torch.cat((zmc_code, mc_image.squeeze(1)), dim=1)
        # # combine
        # zmc_all = self.fc(zmc_all)
        # zmc_all = zmc_all.view(-1, self.gf_dim, 4, 4)
        # h_code = self.upsample1(zmc_all)
        # h_code = self.upsample2(h_code)
        # h_code = self.upsample3(h_code)
        # h_code = self.upsample4(h_code)
        # # state size 3 x 64 x 64
        # h = self.img(h_code)
        # fake_video = h.view(int(h.size(0) / self.video_len), self.video_len, self.n_channels, h.size(3), h.size(3))
        # fake_video = fake_video.permute(0, 2, 1, 3, 4)
        # return None, fake_video, m_mu, m_logvar, r_mu, r_logvar

        # Moda: here comes CP-CSV  (NoCascade) code (the replacement). It is with Extra return wrt DuCo original code
        zmc_all_ = torch.cat((zmc_code, mc_image.squeeze(1)), dim = 1)
        zmc_img = self.fc(zmc_all_).view(-1, self.gf_dim, 4, 4)

        if self.use_segment:
            zmc_seg = self.fc_seg(zmc_all_).view(-1, self.gf_dim_seg, 4, 4)
            zmc_img = self.seg_c(zmc_seg) * zmc_img + zmc_img

            h_seg = self.upsample1_seg(zmc_seg)
            h_img = self.upsample1(zmc_img)
            h_img = self.seg_c1(h_seg) * h_img + h_img

            h_seg = self.upsample2_seg(h_seg)
            h_img = self.upsample2(h_img)
            # h_img = self.seg_c2(h_seg) * h_img + h_img

            h_seg = self.upsample3_seg(h_seg)
            h_img = self.upsample3(h_img)
            # h_img = self.seg_c3(h_seg) * h_img + h_img

            h_seg = self.upsample4_seg(h_seg)
            h_img = self.upsample4(h_img)
            # h_img = self.seg_c4(h_seg) * h_img + h_img
            # generate seg
            segm_video = self.img_seg(h_seg)
            segm_temp = segm_video.view(-1, self.video_len, 1, self.segment_size, self.segment_size)
            segm_temp = segm_temp.permute(0, 2, 1, 3, 4)
            # generate video
            fake_video = self.img(h_img)
            fake_video = fake_video.view(-1, self.video_len, self.n_channels, self.segment_size, self.segment_size)
            fake_video = fake_video.permute(0, 2, 1, 3, 4)

            if seg==True:
                return None, fake_video,  m_mu, m_logvar, r_mu, r_logvar, segm_video # m_mu(60,365), m_logvar(60,365), r_mu(12,124), r_logvar(12,124)
            else:
                return None, fake_video,  m_mu, m_logvar, r_mu, r_logvar, None # m_mu(60,365), m_logvar(60,365), r_mu(12,124), r_logvar(12,124)
        else:
            h_code = self.upsample1(zmc_img) # h_code: batch_size*video_len, 1024, 8, 8 *
            h_code = self.upsample2(h_code)  # h_code: batch_size*video_len, 512, 16, 16 *
            h_code = self.upsample3(h_code)  # h_code: batch_size*video_len, 256, 32, 32 *
            h_code = self.upsample4(h_code)  # h_code: batch_size*video_len=60, 128, 64, 64 *
            # state size 3 x 64 x 64
            h = self.img(h_code) ## *
            fake_video = h.view( int(h.size(0)/self.video_len), self.video_len, self.n_channels, h.size(3), h.size(3)) # 12, 5, 3, 64, 64
            fake_video = fake_video.permute(0, 2, 1, 3, 4) # 12, 3, 5, 64, 64
            #pdb.set_trace()
            return None, fake_video,  m_mu, m_logvar, r_mu, r_logvar, None # m_mu(60,365), m_logvar(60,365), r_mu(12,124), r_logvar(12,124)

    def sample_images(self, motion_input, content_input, seg=False): # Moda: Seg flag is added, also extra return
        m_code, m_mu, m_logvar = motion_input, motion_input, motion_input  # self.ca_net(motion_input)
        # print(content_input.shape, cfg.VIDEO_LEN)
        # content_input = content_input.view(-1, cfg.VIDEO_LEN * content_input.shape[2])
        content_input = torch.reshape(content_input, (-1, cfg.VIDEO_LEN * content_input.shape[2]))
        c_code, c_mu, c_logvar = self.ca_net(content_input) ## h0
        crnn_code, _, _ = self.motion_content_rnn(motion_input, c_mu) ## GRU # Moda: add dummy return
        # Moda: video_len=1 is added to match the updated function with extra parameter
        zm_code = self.sample_z_motion(m_code, 1) ## Text2Gist)
        # one
        zmc_code = torch.cat((zm_code, c_mu), dim=1) # (60,365 ; 60,124)->(60,489)
        # two
        m_image = self.image_net(m_code) ## *
        m_image = m_image.view(-1, self.filter_num, self.image_size) #(60,3,124)
        c_filter = self.filter_net(crnn_code) ## *
        c_filter = c_filter.view(-1, self.out_num, self.filter_num, self.filter_size)
        mc_image = self.dfn_layer([m_image, c_filter]) ## * #(60,1,124)

        # Moda: next commented code is DuCo code, it is replaced with CP-CSV  (NoCascade) c_code
        # bcs it contains it in the else branch. But becarful, this function has extra return now
        # zmc_all = torch.cat((zmc_code, mc_image.squeeze(1)), dim=1)
        # # combine
        # zmc_all = self.fc(zmc_all)
        # zmc_all = zmc_all.view(-1, self.gf_dim, 4, 4)
        # h_code = self.upsample1(zmc_all)
        # h_code = self.upsample2(h_code)
        # h_code = self.upsample3(h_code)
        # h_code = self.upsample4(h_code)
        # # state size 3 x 64 x 64
        # fake_img = self.img(h_code)
        # return None, fake_img, m_mu, m_logvar, c_mu, c_logvar

        # Moda: here comes CP-CSV  (NoCascade) code (the replacement). It is with Extra return wrt DuCo original code
        zmc_all_ = torch.cat((zmc_code, mc_image.squeeze(1)), dim = 1) # (60,613)

        zmc_img = self.fc(zmc_all_).view(-1, self.gf_dim, 4, 4)
        if self.use_segment:
            zmc_seg = self.fc_seg(zmc_all_).view(-1, self.gf_dim_seg, 4, 4)
            zmc_img = self.seg_c(zmc_seg) * zmc_img + zmc_img

            h_seg = self.upsample1_seg(zmc_seg)
            h_img = self.upsample1(zmc_img)
            h_img = self.seg_c1(h_seg) * h_img + h_img

            h_seg = self.upsample2_seg(h_seg)
            h_img = self.upsample2(h_img)
            # h_img = self.seg_c2(h_seg) * h_img + h_img

            h_seg = self.upsample3_seg(h_seg)
            h_img = self.upsample3(h_img)
            # h_img = self.seg_c3(h_seg) * h_img + h_img

            h_seg = self.upsample4_seg(h_seg)
            h_img = self.upsample4(h_img)
            # h_img = self.seg_c4(h_seg) * h_img + h_img

            # generate seg
            segm_img = self.img_seg(h_seg)
            # generatte video
            fake_img = self.img(h_img)
            fake_img = fake_img.view(-1, self.n_channels, self.segment_size, self.segment_size)

            if seg==True:
                return None, fake_img, m_mu, m_logvar, c_mu, c_logvar, segm_img
            else:
                return None, fake_img, m_mu, m_logvar, c_mu, c_logvar, None
        else:
            h_code = self.upsample1(zmc_img) ## *
            h_code = self.upsample2(h_code)  ## *
            h_code = self.upsample3(h_code)  ## *
            h_code = self.upsample4(h_code)  ## *
            # state size 3 x 64 x 64
            fake_img = self.img(h_code)
            return None, fake_img, m_mu, m_logvar, c_mu, c_logvar, None

class MartNoCascade(StoryGAN):
    def define_module(self):

        ninput = self.motion_dim + self.content_dim + self.image_size
        ngf = self.gf_dim

        self.ca_net = CA_NET()
        # -> ngf x 4 x 4
        self.fc = nn.Sequential(
            nn.Linear(ninput, ngf * 4 * 4, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4),
            nn.ReLU(True))

        self.filter_net = nn.Sequential(
            nn.Linear(self.content_dim, self.filter_size * self.filter_num * self.out_num),
            nn.BatchNorm1d(self.filter_size * self.filter_num * self.out_num))

        self.image_net = nn.Sequential(
            nn.Linear(self.motion_dim, self.image_size * self.filter_num),
            nn.BatchNorm1d(self.image_size * self.filter_num),
            nn.Tanh())

        self.mart_fc = nn.Sequential(
            nn.Linear(self.content_dim + self.cfg.LABEL_NUM, self.content_dim),
            nn.BatchNorm1d(self.content_dim),
            nn.Tanh())

        # ngf x 4 x 4 -> ngf/2 x 8 x 8
        self.upsample1 = upBlock(ngf, ngf // 2)
        # -> ngf/4 x 16 x 16
        self.upsample2 = upBlock(ngf // 2, ngf // 4)

        # For two stage
        if self.cfg.TWO_STG:
            # -> ngf/16 x 32 x 32
            self.upsample3 = upBlock(ngf // 4, ngf // 16)
            self.next_g = NEXT_STAGE_G(ngf // 16, self.cfg.MART.hidden_size, self.content_dim, attn_type=self.cfg.TWO_STG_ATTN)
            self.next_img = nn.Sequential(
            conv3x3(ngf // 16, 3),  # -> 3 x 64 x 64
            nn.Tanh())

        else:
            # -> ngf/8 x 32 x 32
            self.upsample3 = upBlock(ngf // 4, ngf // 8)
            # -> ngf/16 x 64 x 64
            self.upsample4 = upBlock(ngf // 8, ngf // 16)
            # -> 3 x 64 x 64
            self.img = nn.Sequential(
                conv3x3(ngf // 16, 3),
                nn.Tanh())

        self.m_net = nn.Sequential(
            nn.Linear(self.motion_dim, self.motion_dim),
            nn.BatchNorm1d(self.motion_dim))

        self.c_net = nn.Sequential(
            nn.Linear(self.content_dim, self.cfg.MART.hidden_size),
            nn.BatchNorm1d(self.cfg.MART.hidden_size))

        self.dfn_layer = DynamicFilterLayer(self.filter_size,
                                            pad=self.filter_size // 2)

        # Moda: Adding Seg part
        if self.use_segment:
            ngf_seg = self.gf_dim_seg

            self.seg_c = conv3x3(ngf_seg, ngf)
            self.seg_c1 = conv3x3(ngf_seg//2, ngf//2)
            # self.seg_c2 = conv3x3(ngf_seg//4, ngf//4)
            # self.seg_c3 = conv3x3(ngf_seg//8, ngf//8)
            # self.seg_c4 = conv3x3(ngf_seg//16, ngf//16)

            # For generate seg and img v4 and v5 and v6
            self.fc_seg = nn.Sequential(
                nn.Linear(ninput, ngf_seg * 4 * 4, bias=False),
                nn.BatchNorm1d(ngf_seg * 4 * 4),
                nn.ReLU(True))
            # ngf x 4 x 4 -> ngf/2 x 8 x 8
            self.upsample1_seg = upBlock(ngf_seg, ngf_seg // 2)
            # -> ngf/4 x 16 x 16
            self.upsample2_seg = upBlock(ngf_seg // 2, ngf_seg // 4)
            # -> ngf/8 x 32 x 32
            self.upsample3_seg = upBlock(ngf_seg // 4, ngf_seg // 8)
            # -> ngf/16 x 64 x 64
            self.upsample4_seg = upBlock(ngf_seg // 8, ngf_seg // 16)
            # -> 3 x 64 x 64
            self.img_seg = nn.Sequential(
                conv3x3(ngf_seg // 16, 1),
                nn.Tanh())

    def sample_videos(self, motion_input, content_input, caption_input_ids=None, caption_input_masks=None, labels=None, seg=False): # Moda: Seg flag is added, also extra return
        content_input = content_input.view(-1, self.cfg.VIDEO_LEN * content_input.shape[2])
        r_code, r_mu, r_logvar = self.ca_net(torch.squeeze(content_input))
        c_mu = r_mu.repeat(self.video_len, 1).view(-1, r_mu.shape[1])

        crnn_code, word_embs, _ = self.motion_content_rnn(caption_input_ids, caption_input_masks, r_code, labels)
        temp = motion_input.view(-1, motion_input.shape[2])
        m_code, m_mu, m_logvar = temp, temp, temp  # self.ca_net(temp)
        m_code = m_code.view(motion_input.shape[0], self.video_len, self.motion_dim)
        zm_code = self.sample_z_motion(m_code, self.video_len)

        # one
        zmc_code = torch.cat((zm_code, c_mu), dim=1)
        # two
        m_image = self.image_net(m_code.view(-1, m_code.shape[2]))

        m_image = m_image.view(-1, self.filter_num, self.image_size)
        c_filter = self.filter_net(crnn_code)
        c_filter = c_filter.view(-1, self.out_num, self.filter_num, self.filter_size)
        mc_image = self.dfn_layer([m_image, c_filter])

        # Moda: next commented code is DuCo code, it is replaced with CP-CSV  (NoCascade) c_code
        # bcs it contains it in the else branch. But becarful, this function has extra return now
        # # INIT_G_START
        # zmc_all = torch.cat((zmc_code, mc_image.squeeze(1)), dim=1)
        # zmc_all = self.fc(zmc_all)
        # zmc_all = zmc_all.view(-1, self.gf_dim, 4, 4)
        # h_code = self.upsample1(zmc_all)
        # h_code = self.upsample2(h_code)
        # h_code = self.upsample3(h_code)
        #
        # ################# NEW #############################
        # # need to fix the dimensions of word_embs
        # # r_code = [b * GAN.CONDITION_DIM]
        # if self.cfg.TWO_STG:
        #     r_code = r_code.unsqueeze(1).repeat(1, self.video_len, 1).view(-1, r_code.shape[-1])
        #     # print(h_code.shape, r_code.shape, word_embs[0].shape)
        #     # h_code2, _ = self.next_g(h_code, r_code, torch.cat(word_embs, dim=0).transpose(-2, -1),
        #     #                          caption_input_masks.view(-1, caption_input_masks.shape[-1]))
        #     seq_len, word_emb_dim = word_embs[0].shape[-2], word_embs[0].shape[-1]
        #     # print(seq_len, word_emb_dim)
        #     # print(torch.stack(word_embs).shape)
        #     # print(torch.stack(word_embs).transpose(1,0).shape)
        #     h_code2, _ = self.next_g(h_code, r_code, torch.stack(word_embs).transpose(1, 0).reshape(-1, seq_len, word_emb_dim).transpose(-2, -1),
        #                              caption_input_masks.view(-1, caption_input_masks.shape[-1]))
        #     h = self.next_img(h_code2)
        # else:
        #     h_code = self.upsample4(h_code)
        #     # INIT_G_END
        #     # state size 3 x 64 x 64
        #     # GET_IMG_G_START
        #     h = self.img(h_code)
        #     # GET_IMG_G_END
        # ################ NEW ###############################
        #
        # fake_video = h.view(int(h.size(0) / self.video_len), self.video_len, self.n_channels, h.size(3), h.size(3))
        # fake_video = fake_video.permute(0, 2, 1, 3, 4)
        # # print("Generated video shape", fake_video.shape)
        # return None, fake_video, m_mu, m_logvar, r_mu, r_logvar


        # Moda: here comes CP-CSV  (NoCascade) code (the replacement). It is with Extra return wrt DuCo original code
        # DuCo MART implementation is added in both brnches of use_segment
        # Basically, it replaces the upsample4 part
        zmc_all_ = torch.cat((zmc_code, mc_image.squeeze(1)), dim = 1)
        zmc_img = self.fc(zmc_all_).view(-1, self.gf_dim, 4, 4)

        if self.use_segment:
            zmc_seg = self.fc_seg(zmc_all_).view(-1, self.gf_dim_seg, 4, 4)
            zmc_img = self.seg_c(zmc_seg) * zmc_img + zmc_img

            h_seg = self.upsample1_seg(zmc_seg)
            h_img = self.upsample1(zmc_img)
            h_img = self.seg_c1(h_seg) * h_img + h_img

            h_seg = self.upsample2_seg(h_seg)
            h_img = self.upsample2(h_img)
            # h_img = self.seg_c2(h_seg) * h_img + h_img

            h_seg = self.upsample3_seg(h_seg)
            h_img = self.upsample3(h_img)
            # h_img = self.seg_c3(h_seg) * h_img + h_img

            # Moda: MART implememntation
            ## TODO: make sure next_g & next_img are applicable for segmentation also
            ################# NEW #############################
            # need to fix the dimensions of word_embs
            # r_code = [b * GAN.CONDITION_DIM]
            if self.cfg.TWO_STG:
                r_code = r_code.unsqueeze(1).repeat(1, self.video_len, 1).view(-1, r_code.shape[-1])
                # print(h_code.shape, r_code.shape, word_embs[0].shape)
                # h_code2, _ = self.next_g(h_code, r_code, torch.cat(word_embs, dim=0).transpose(-2, -1),
                #                          caption_input_masks.view(-1, caption_input_masks.shape[-1]))
                seq_len, word_emb_dim = word_embs[0].shape[-2], word_embs[0].shape[-1]
                # print(seq_len, word_emb_dim)
                # print(torch.stack(word_embs).shape)
                # print(torch.stack(word_embs).transpose(1,0).shape)

                # Moda: (The same as in else branch)
                # 1- h_code2 is renamed as h_img2
                # 2- h is renamed as fake_video
                # 3- Seg video is added
                # 4- Check Next "Moda" comment ...
                h_seg2, _ = self.next_g(h_seg, r_code, torch.stack(word_embs).transpose(1, 0).reshape(-1, seq_len, word_emb_dim).transpose(-2, -1),
                                         caption_input_masks.view(-1, caption_input_masks.shape[-1]))
                segm_video = self.next_img(h_seg2)

                h_img2, _ = self.next_g(h_img, r_code, torch.stack(word_embs).transpose(1, 0).reshape(-1, seq_len, word_emb_dim).transpose(-2, -1),
                                         caption_input_masks.view(-1, caption_input_masks.shape[-1]))
                fake_video = self.next_img(h_img2)
            else:
                # Moda: check modifications in the prev comment
                h_seg = self.upsample4(h_seg)
                h_img = self.upsample4(h_img)
                # INIT_G_END
                # state size 3 x 64 x 64
                # GET_IMG_G_START
                segm_video = self.img_seg(h_seg)
                fake_video = self.img(h_img)
                # GET_IMG_G_END

            ################ NEW ###############################
            # Moda: The next two lines are commented, bcs they are replaced by the prev "TWO_STG" condition (MART)
            # h_seg = self.upsample4_seg(h_seg) # Moda
            # h_img = self.upsample4(h_img) # Moda

            # h_img = self.seg_c4(h_seg) * h_img + h_img

            # generate seg
            # Moda: The next line is commented, bcs it's replaced by the prev "TWO_STG" condition (MART)
            # segm_video = self.img_seg(h_seg) # Moda
            segm_temp = segm_video.view(-1, self.video_len, 1, self.segment_size, self.segment_size)
            segm_temp = segm_temp.permute(0, 2, 1, 3, 4)
            # generate video
            # Moda: The next line is commented, bcs it's replaced by the prev "TWO_STG" condition (MART)
            # fake_video = self.img(h_img) # Moda
            fake_video = fake_video.view(-1, self.video_len, self.n_channels, self.segment_size, self.segment_size)
            fake_video = fake_video.permute(0, 2, 1, 3, 4)

            if seg==True:
                return None, fake_video, m_mu, m_logvar, r_mu, r_logvar, segm_video # m_mu(60,365), m_logvar(60,365), r_mu(12,124), r_logvar(12,124)
            else:
                return None, fake_video, m_mu, m_logvar, r_mu, r_logvar, None # m_mu(60,365), m_logvar(60,365), r_mu(12,124), r_logvar(12,124)
        else:
            h_code = self.upsample1(zmc_img) # h_code: batch_size*video_len, 1024, 8, 8 *
            h_code = self.upsample2(h_code)  # h_code: batch_size*video_len, 512, 16, 16 *
            h_code = self.upsample3(h_code)  # h_code: batch_size*video_len, 256, 32, 32 *

            # Moda: MART implememntation
            if self.cfg.TWO_STG:
                r_code = r_code.unsqueeze(1).repeat(1, self.video_len, 1).view(-1, r_code.shape[-1])
                # print(h_code.shape, r_code.shape, word_embs[0].shape)
                # h_code2, _ = self.next_g(h_code, r_code, torch.cat(word_embs, dim=0).transpose(-2, -1),
                #                          caption_input_masks.view(-1, caption_input_masks.shape[-1]))
                seq_len, word_emb_dim = word_embs[0].shape[-2], word_embs[0].shape[-1]
                # print(seq_len, word_emb_dim)
                # print(torch.stack(word_embs).shape)
                # print(torch.stack(word_embs).transpose(1,0).shape)
                h_code2, _ = self.next_g(h_code, r_code, torch.stack(word_embs).transpose(1, 0).reshape(-1, seq_len, word_emb_dim).transpose(-2, -1),
                                         caption_input_masks.view(-1, caption_input_masks.shape[-1]))
                h = self.next_img(h_code2)
            else:
                # Moda: the last step of h_code is inside the else
                h_code = self.upsample4(h_code)  # h_code: batch_size*video_len=60, 128, 64, 64 *
                # state size 3 x 64 x 64
                h = self.img(h_code) ## *
            fake_video = h.view( int(h.size(0)/self.video_len), self.video_len, self.n_channels, h.size(3), h.size(3)) # 12, 5, 3, 64, 64
            fake_video = fake_video.permute(0, 2, 1, 3, 4) # 12, 3, 5, 64, 64
            #pdb.set_trace()
            return None, fake_video,  m_mu, m_logvar, r_mu, r_logvar, None # m_mu(60,365), m_logvar(60,365), r_mu(12,124), r_logvar(12,124)

    def sample_images(self, motion_input, content_input, caption_input_ids=None, caption_input_masks=None, labels=None, seg=False): # Moda: Seg flag is added, also extra return
        m_code, m_mu, m_logvar = motion_input, motion_input, motion_input  # self.ca_net(motion_input)
        # print(content_input.shape, cfg.VIDEO_LEN)
        # content_input = content_input.view(-1, cfg.VIDEO_LEN * content_input.shape[2])
        content_input = torch.reshape(content_input, (-1, self.cfg.VIDEO_LEN * content_input.shape[2]))
        c_code, c_mu, c_logvar = self.ca_net(content_input)
        crnn_code, word_embs, _ = self.motion_content_rnn(caption_input_ids.unsqueeze(1), caption_input_masks.unsqueeze(1), c_mu, labels.unsqueeze(1)) # wut

        zm_code = self.sample_z_motion(m_code, 1)
        # one
        zmc_code = torch.cat((zm_code, c_mu), dim=1)
        # two
        m_image = self.image_net(m_code)
        m_image = m_image.view(-1, self.filter_num, self.image_size)
        c_filter = self.filter_net(crnn_code)
        c_filter = c_filter.view(-1, self.out_num, self.filter_num, self.filter_size)

        mc_image = self.dfn_layer([m_image, c_filter])


        # Moda: next commented code is DuCo code, it is replaced with CP-CSV  (NoCascade) c_code
        # bcs it contains it in the else branch. But becarful, this function has extra return now
        # zmc_all = torch.cat((zmc_code, mc_image.squeeze(1)), dim=1)
        # # combine
        # zmc_all = self.fc(zmc_all)
        # zmc_all = zmc_all.view(-1, self.gf_dim, 4, 4)
        # h_code = self.upsample1(zmc_all)
        # h_code = self.upsample2(h_code)
        # h_code = self.upsample3(h_code)
        #
        # if self.cfg.TWO_STG:
        #     h_code2, _ = self.next_g(h_code, c_code, word_embs[0].transpose(-2, -1), caption_input_masks)
        #     fake_img = self.next_img(h_code2)
        # else:
        #     h_code = self.upsample4(h_code)
        #     # state size 3 x 64 x 64
        #     fake_img = self.img(h_code)
        #
        # # print("Generated image shape", fake_img.shape)
        #
        # return None, fake_img, m_mu, m_logvar, c_mu, c_logvar

        # Moda: here comes CP-CSV  (NoCascade) code (the replacement). It is with Extra return wrt DuCo original code
        # DuCo MART implementation is added in both brnches of use_segment
        # Basically, it replaces the upsample4 part
        zmc_all_ = torch.cat((zmc_code, mc_image.squeeze(1)), dim = 1)
        zmc_img = self.fc(zmc_all_).view(-1, self.gf_dim, 4, 4)

        if self.use_segment:
            zmc_seg = self.fc_seg(zmc_all_).view(-1, self.gf_dim_seg, 4, 4)
            zmc_img = self.seg_c(zmc_seg) * zmc_img + zmc_img

            h_seg = self.upsample1_seg(zmc_seg)
            h_img = self.upsample1(zmc_img)
            h_img = self.seg_c1(h_seg) * h_img + h_img

            h_seg = self.upsample2_seg(h_seg)
            h_img = self.upsample2(h_img)
            # h_img = self.seg_c2(h_seg) * h_img + h_img

            h_seg = self.upsample3_seg(h_seg)
            h_img = self.upsample3(h_img)
            # h_img = self.seg_c3(h_seg) * h_img + h_img

            # Moda: MART implememntation
            ## TODO: make sure that next_g & next_img are applicable for segmentation also
            ################# NEW #############################
            if self.cfg.TWO_STG:
                h_seg2, _ = self.next_g(h_seg, c_code, word_embs[0].transpose(-2, -1), caption_input_masks)
                segm_img = self.next_img(h_seg2)

                h_img2, _ = self.next_g(h_img, c_code, word_embs[0].transpose(-2, -1), caption_input_masks)
                fake_img = self.next_img(h_img2)
            else:
                # Moda: check modifications in the prev comment
                h_seg = self.upsample4(h_seg)
                h_img = self.upsample4(h_img)
                # INIT_G_END
                # state size 3 x 64 x 64
                # GET_IMG_G_START
                segm_img = self.img_seg(h_seg)
                fake_img = self.img(h_img)
                # GET_IMG_G_END
            ################ NEW ###############################
            # Moda: The next two lines are commented, bcs they are replaced by the prev "TWO_STG" condition (MART)
            # h_seg = self.upsample4_seg(h_seg) # Moda
            # h_img = self.upsample4(h_img) # Moda

            # h_img = self.seg_c4(h_seg) * h_img + h_img

            if seg==True:
                return None, fake_img, m_mu, m_logvar, c_mu, c_logvar, segm_img # m_mu(60,365), m_logvar(60,365), r_mu(12,124), r_logvar(12,124)
            else:
                return None, fake_img, m_mu, m_logvar, c_mu, c_logvar, None
        else:
            h_code = self.upsample1(zmc_img) ## *
            h_code = self.upsample2(h_code)  ## *
            h_code = self.upsample3(h_code)  ## *

            # Moda: MART implememntation
            if self.cfg.TWO_STG:
                h_code2, _ = self.next_g(h_code, c_code, word_embs[0].transpose(-2, -1), caption_input_masks)
                fake_img = self.next_img(h_code2)
            else:
                # Moda: the last step of h_code is inside the else
                h_code = self.upsample4(h_code)
                # state size 3 x 64 x 64
                fake_img = self.img(h_code)
            return None, fake_img, m_mu, m_logvar, c_mu, c_logvar, None
    
    def motion_content_rnn(self, word_input_ids_list, input_masks_list, c_code, labels, return_memory=False): # Moda: over write NoMART method
        """
        Args:
            input_ids_list: [(N, L)] * step_size
            input_masks_list: [(N, L)] * step_size with 1 indicates valid bits
                will not be used when return_memory is True, thus can be None in this case
            return_memory: bool,
        Returns:
        """
        # [(N, M, D)] * num_hidden_layers, initialized internally

        c_code = self.c_net(c_code)

        memory_list = []
        memory_list.append([c_code.unsqueeze(1).repeat(1, self.cfg.MART.n_memory_cells, 1), c_code.unsqueeze(1).repeat(1, self.cfg.MART.n_memory_cells, 1)])
        video_len = word_input_ids_list.shape[1]

        encoded_outputs_list = []  # [(N, L, D)] * step_size
        for idx in range(video_len):
            ms, encoded_layer_outputs = self.mart_forward_step(memory_list[-1],
                                                               word_input_ids_list[:, idx, :],
                                                               input_masks_list[:, idx, :])

            memory_list.append(ms)
            encoded_outputs_list.append(encoded_layer_outputs[-1])

        c_m_t = [self.pooler(h_k, input_masks_list[:, idx, :]).unsqueeze(1) for idx, h_k in enumerate(encoded_outputs_list)]
        mocornn_co = self.mart_fc(torch.cat((torch.cat(c_m_t, dim=1), labels), dim=-1).view(-1, self.content_dim + self.cfg.LABEL_NUM))

        if return_memory:  # used to analyze memory
            return mocornn_co, encoded_outputs_list, memory_list
        else:  # normal training/evaluation mode
            return mocornn_co, encoded_outputs_list, None

class NoMartCascade(StoryGAN):
    def define_module(self):
        ninput = self.motion_dim + self.content_dim + self.image_size # (365+124+124=613)
        ngf = self.gf_dim # 128*8=1024

        self.ca_net = CA_NET()
        # -> ngf x 4 x 4

        self.filter_net = nn.Sequential(
            nn.Linear(self.content_dim, self.filter_size * self.filter_num * self.out_num),
            nn.BatchNorm1d(self.filter_size * self.filter_num * self.out_num))

        self.image_net = nn.Sequential(
            nn.Linear(self.motion_dim, self.image_size * self.filter_num),
            nn.BatchNorm1d(self.image_size * self.filter_num),
            nn.Tanh())

        # For generate final image
        self.fc = nn.Sequential(
            nn.Linear(ninput, ngf * 4 * 4, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4),
            nn.ReLU(True))
        # ngf x 4 x 4 -> ngf/2 x 8 x 8
        self.upsample1 = upBlock(ngf, ngf // 2)
        # -> ngf/4 x 16 x 16
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        # -> ngf/8 x 32 x 32
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        # -> ngf/16 x 64 x 64
        self.upsample4 = upBlock(ngf // 8, ngf // 16)
        # -> 3 x 64 x 64
        self.img = nn.Sequential(
            conv3x3(ngf // 16, 3),
            nn.Tanh())
		# Moda: Adding Seg part
        if self.use_segment:
            ngf_seg = self.gf_dim_seg

            self.seg_c = conv3x3(ngf_seg, ngf) # 2048, 256
            self.seg_c1 = conv3x3(ngf_seg//2, ngf//2) # 1024, 128
            # self.seg_c2 = conv3x3(ngf_seg//4, ngf//4)
            # self.seg_c3 = conv3x3(ngf_seg//8, ngf//8)
            # self.seg_c4 = conv3x3(ngf_seg//16, ngf//16)

            # For generate seg and img v4 and v5 and v6
            self.fc_seg = nn.Sequential(
                nn.Linear(ninput, ngf_seg * 4 * 4, bias=False),
                nn.BatchNorm1d(ngf_seg * 4 * 4),
                nn.ReLU(True))
            # ngf x 4 x 4 -> ngf/2 x 8 x 8
            self.upsample1_seg = upBlock(ngf_seg, ngf_seg // 2)
            # -> ngf/4 x 16 x 16
            self.upsample2_seg = upBlock(ngf_seg // 2, ngf_seg // 4)
            # -> ngf/8 x 32 x 32
            self.upsample3_seg = upBlock(ngf_seg // 4, ngf_seg // 8)
            # -> ngf/16 x 64 x 64
            self.upsample4_seg = upBlock(ngf_seg // 8, ngf_seg // 16)
            # -> 1 x 64 x 64
            self.img_seg = nn.Sequential(
                conv3x3(ngf_seg // 16, 1),
                nn.Tanh())
            self.presample = nn.Sequential(
                conv3x3(1, ngf_seg // 16),
                nn.BatchNorm2d(ngf_seg // 16),
                nn.ReLU(),
            )
            self.downsample1_seg = downBlock( ngf_seg // 16, ngf_seg // 8 )
            self.downsample2_seg = downBlock( ngf_seg // 8, ngf_seg // 4 )
            self.downsample3_seg = downBlock( ngf_seg // 4, ngf_seg // 2 )
            self.downsample4_seg = downBlock( ngf_seg // 2, ngf_seg )

        self.m_net = nn.Sequential(
            nn.Linear(self.motion_dim, self.motion_dim),
            nn.BatchNorm1d(self.motion_dim))

        self.c_net = nn.Sequential(
            nn.Linear(self.content_dim, self.content_dim),
            nn.BatchNorm1d(self.content_dim))

        self.dfn_layer = DynamicFilterLayer(self.filter_size,
                                            pad=self.filter_size // 2)

    def sample_videos(self, motion_input, content_input, seg=False): # Moda: Seg flag is added, also extra return
        # Moda: return changed bcs of CP-CSV (Cascade)
		###
        # motion_input:  batch_size, video_len, 365
        # content_input: batch_size, video_len, 356
        ###
        content_input = content_input.view(-1, cfg.VIDEO_LEN * content_input.shape[2])
        # Moda: the next commented part is from CP-CSV  (NoCascade) (might be needed). Squeezing content_input based on a condition
        # if content_input.shape[0] > 1:
        #     content_input = torch.squeeze(content_input)
        r_code, r_mu, r_logvar = self.ca_net(torch.squeeze(content_input)) ## h0
        c_mu = r_mu.repeat(self.video_len, 1).view(-1, r_mu.shape[1])

        crnn_code, _, _ = self.motion_content_rnn(motion_input, r_code) ## i_t = GRU(s_t) # Moda: add dummy return

        temp = motion_input.view(-1, motion_input.shape[2])
        m_code, m_mu, m_logvar = temp, temp, temp  # self.ca_net(temp)
        m_code = m_code.view(motion_input.shape[0], self.video_len, self.motion_dim)
        zm_code = self.sample_z_motion(m_code) ## *

        # one
        zmc_code = torch.cat((zm_code, c_mu), dim=1)
        # two
        m_image = self.image_net(m_code.view(-1, m_code.shape[2])) ## linearly transform motion(365) to image(372)
        m_image = m_image.view(-1, self.filter_num, self.image_size)
        c_filter = self.filter_net(crnn_code) ## Filter(i_t)
        c_filter = c_filter.view(-1, self.out_num, self.filter_num, self.filter_size)
        mc_image = self.dfn_layer([m_image, c_filter]) ## *

        # Moda: next commented code is DuCo code, it is replaced with CP-CSV  (NoCascade) c_code
        # bcs it contains it in the else branch. But becarful, this function has extra return now
        # zmc_all = torch.cat((zmc_code, mc_image.squeeze(1)), dim=1)
        # # combine
        # zmc_all = self.fc(zmc_all)
        # zmc_all = zmc_all.view(-1, self.gf_dim, 4, 4)
        # h_code = self.upsample1(zmc_all)
        # h_code = self.upsample2(h_code)
        # h_code = self.upsample3(h_code)
        # h_code = self.upsample4(h_code)
        # # state size 3 x 64 x 64
        # h = self.img(h_code)
        # fake_video = h.view(int(h.size(0) / self.video_len), self.video_len, self.n_channels, h.size(3), h.size(3))
        # fake_video = fake_video.permute(0, 2, 1, 3, 4)
        # return None, fake_video, m_mu, m_logvar, r_mu, r_logvar

        # Moda: here comes CP-CSV  (NoCascade) code (the replacement). It is with Extra return wrt DuCo original code
        zmc_all_ = torch.cat((zmc_code, mc_image.squeeze(1)), dim = 1)
        zmc_img = self.fc(zmc_all_).view(-1, self.gf_dim, 4, 4)

        if self.use_segment: # Moda: this branch changed as per CP-CSV (Cascade)
            zmc_seg = self.fc_seg(zmc_all_).view(-1, self.gf_dim_seg, 4, 4)
            h_seg1 = self.upsample1_seg(zmc_seg) #;print(h_seg1.shape)
            h_seg2 = self.upsample2_seg(h_seg1) #;print(h_seg2.shape)
            h_seg3 = self.upsample3_seg(h_seg2) #;print(h_seg3.shape)
            h_seg4 = self.upsample4_seg(h_seg3) #;print(h_seg4.shape)

            # generate seg
            segm_video = self.img_seg(h_seg4)

            zmc_latent = self.presample(segm_video) #;print(zmc_latent.shape)
            g_seg4 = self.downsample1_seg(zmc_latent) #;print(g_seg4.shape)
            g_seg3 = self.downsample2_seg(g_seg4) #;print(g_seg3.shape)
            g_seg2 = self.downsample3_seg(g_seg3) #;print(g_seg2.shape)
            g_seg1 = self.downsample4_seg(g_seg2) #;print(g_seg1.shape)


            zmc_img = self.seg_c(g_seg1) * zmc_img + zmc_img


            h_img = self.upsample1(zmc_img)
            h_img = self.seg_c1(g_seg2) * h_img + h_img # batch_size*video_len, 1024, 8, 8

            h_img = self.upsample2(h_img)
            # h_img = self.seg_c2(g_seg3) * h_img + h_img

            h_img = self.upsample3(h_img)
            # h_img = self.seg_c3(g_seg4) * h_img + h_img

            h_img = self.upsample4(h_img)
            # h_img = self.seg_c4(zmc_latent) * h_img + h_img
            segm_temp = segm_video.view(-1, self.video_len, 1, self.segment_size, self.segment_size)
            segm_temp = segm_temp.permute(0, 2, 1, 3, 4)
            # generate video
            fake_video = self.img(h_img)
            fake_video = fake_video.view(-1, self.video_len, self.n_channels, self.segment_size, self.segment_size)
            fake_video = fake_video.permute(0, 2, 1, 3, 4)

            if seg==True:
                return ((zmc_seg, h_seg1, h_seg2, h_seg3), (g_seg1, g_seg2, g_seg3, g_seg4)), \
                    fake_video, m_mu, m_logvar, r_mu, r_logvar, segm_video # m_mu(60,365), m_logvar(60,365), r_mu(12,124), r_logvar(12,124)
            else:
                return ((zmc_seg, h_seg1, h_seg2, h_seg3), (g_seg1, g_seg2, g_seg3, g_seg4)), \
                    fake_video, m_mu, m_logvar, r_mu, r_logvar, None # m_mu(60,365), m_logvar(60,365), r_mu(12,124), r_logvar(12,124)
        else:
            h_code = self.upsample1(zmc_img) # h_code: batch_size*video_len, 1024, 8, 8 *
            h_code = self.upsample2(h_code)  # h_code: batch_size*video_len, 512, 16, 16 *
            h_code = self.upsample3(h_code)  # h_code: batch_size*video_len, 256, 32, 32 *
            h_code = self.upsample4(h_code)  # h_code: batch_size*video_len=60, 128, 64, 64 *
            # state size 3 x 64 x 64
            h = self.img(h_code) ## *
            fake_video = h.view( int(h.size(0)/self.video_len), self.video_len, self.n_channels, h.size(3), h.size(3)) # 12, 5, 3, 64, 64
            fake_video = fake_video.permute(0, 2, 1, 3, 4) # 12, 3, 5, 64, 64
            #pdb.set_trace()
            return None, fake_video, m_mu, m_logvar, r_mu, r_logvar, None # m_mu(60,365), m_logvar(60,365), r_mu(12,124), r_logvar(12,124)

    def sample_images(self, motion_input, content_input, seg=False): # Moda: Seg flag is added, also extra return
        m_code, m_mu, m_logvar = motion_input, motion_input, motion_input  # self.ca_net(motion_input)
        # print(content_input.shape, cfg.VIDEO_LEN)
        # content_input = content_input.view(-1, cfg.VIDEO_LEN * content_input.shape[2])
        content_input = torch.reshape(content_input, (-1, cfg.VIDEO_LEN * content_input.shape[2]))
        c_code, c_mu, c_logvar = self.ca_net(content_input) ## h0
        crnn_code, _, _ = self.motion_content_rnn(motion_input, c_mu) ## GRU # Moda: add dummy return
        # Moda: video_len=1 is added to match the updated function with extra parameter
        zm_code = self.sample_z_motion(m_code, 1) ## Text2Gist)
        # one
        zmc_code = torch.cat((zm_code, c_mu), dim=1) # (60,365 ; 60,124)->(60,489)
        # two
        m_image = self.image_net(m_code) ## *
        m_image = m_image.view(-1, self.filter_num, self.image_size) #(60,3,124)
        c_filter = self.filter_net(crnn_code) ## *
        c_filter = c_filter.view(-1, self.out_num, self.filter_num, self.filter_size)
        mc_image = self.dfn_layer([m_image, c_filter]) ## * #(60,1,124)

        # Moda: next commented code is DuCo code, it is replaced with CP-CSV  (NoCascade) c_code
        # bcs it contains it in the else branch. But becarful, this function has extra return now
        # zmc_all = torch.cat((zmc_code, mc_image.squeeze(1)), dim=1)
        # # combine
        # zmc_all = self.fc(zmc_all)
        # zmc_all = zmc_all.view(-1, self.gf_dim, 4, 4)
        # h_code = self.upsample1(zmc_all)
        # h_code = self.upsample2(h_code)
        # h_code = self.upsample3(h_code)
        # h_code = self.upsample4(h_code)
        # # state size 3 x 64 x 64
        # fake_img = self.img(h_code)
        # return None, fake_img, m_mu, m_logvar, c_mu, c_logvar

        # Moda: here comes CP-CSV  (NoCascade) code (the replacement). It is with Extra return wrt DuCo original code
        zmc_all_ = torch.cat((zmc_code, mc_image.squeeze(1)), dim = 1) # (60,613)

        zmc_img = self.fc(zmc_all_).view(-1, self.gf_dim, 4, 4)
        if self.use_segment:
            zmc_seg = self.fc_seg(zmc_all_).view(-1, self.gf_dim_seg, 4, 4)
            # print(zmc_seg.shape)
            h_seg1 = self.upsample1_seg(zmc_seg) #;print(h_seg1.shape)
            h_seg2 = self.upsample2_seg(h_seg1) #;print(h_seg2.shape)
            h_seg3 = self.upsample3_seg(h_seg2) #;print(h_seg3.shape)
            h_seg4 = self.upsample4_seg(h_seg3) #;print(h_seg4.shape)

            # generate seg
            segm_img = self.img_seg(h_seg4)

            zmc_latent = self.presample(segm_img) #;print(zmc_latent.shape)
            g_seg4 = self.downsample1_seg(zmc_latent) #;print(g_seg4.shape)
            g_seg3 = self.downsample2_seg(g_seg4) #;print(g_seg3.shape)
            g_seg2 = self.downsample3_seg(g_seg3) #;print(g_seg2.shape)
            g_seg1 = self.downsample4_seg(g_seg2) #;print(g_seg1.shape)


            zmc_img = self.seg_c(g_seg1) * zmc_img + zmc_img

            h_img = self.upsample1(zmc_img)
            h_img = self.seg_c1(g_seg2) * h_img + h_img # batch_size*video_len, 1024, 8, 8

            h_img = self.upsample2(h_img)
            # h_img = self.seg_c2(g_seg3) * h_img + h_img

            h_img = self.upsample3(h_img)
            # h_img = self.seg_c3(g_seg4) * h_img + h_img
            h_img = self.upsample4(h_img)

            # generatte video
            fake_img = self.img(h_img)
            fake_img = fake_img.view(-1, self.n_channels, self.segment_size, self.segment_size)

            if seg==True:
                return ((zmc_seg, h_seg1, h_seg2, h_seg3), (g_seg1, g_seg2, g_seg3, g_seg4)), \
                    fake_img, m_mu, m_logvar, c_mu, c_logvar, segm_img
            else:
                return ((zmc_seg, h_seg1, h_seg2, h_seg3), (g_seg1, g_seg2, g_seg3, g_seg4)), \
                         fake_img, m_mu, m_logvar, c_mu, c_logvar, None
        else:
            h_code = self.upsample1(zmc_img) ## *
            h_code = self.upsample2(h_code)  ## *
            h_code = self.upsample3(h_code)  ## *
            h_code = self.upsample4(h_code)  ## *
            # state size 3 x 64 x 64
            fake_img = self.img(h_code)
            return None, fake_img, m_mu, m_logvar, c_mu, c_logvar, None

    def train_autoencoder(self, real_segments):
        zmc_latent = self.presample(real_segments) #;print(zmc_latent.shape)
        g_seg4 = self.downsample1_seg(zmc_latent) #;print(g_seg4.shape)
        g_seg3 = self.downsample2_seg(g_seg4) #;print(g_seg3.shape)
        g_seg2 = self.downsample3_seg(g_seg3) #;print(g_seg2.shape)
        g_seg1 = self.downsample4_seg(g_seg2) #;print(g_seg1.shape)

        h_seg1 = self.upsample1_seg(g_seg1) #;print(h_seg1.shape)
        h_seg2 = self.upsample2_seg(h_seg1) #;print(h_seg2.shape)
        h_seg3 = self.upsample3_seg(h_seg2) #;print(h_seg3.shape)
        h_seg4 = self.upsample4_seg(h_seg3) #;print(h_seg4.shape)
        segm_img = self.img_seg(h_seg4)
        return segm_img #(( h_seg1, h_seg2, h_seg3, h_seg4), (g_seg1, g_seg2, g_seg3, g_seg4))

class MartCascade(StoryGAN):
    def define_module(self):

        ninput = self.motion_dim + self.content_dim + self.image_size
        ngf = self.gf_dim

        self.ca_net = CA_NET()
        # -> ngf x 4 x 4
        self.fc = nn.Sequential(
            nn.Linear(ninput, ngf * 4 * 4, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4),
            nn.ReLU(True))

        self.filter_net = nn.Sequential(
            nn.Linear(self.content_dim, self.filter_size * self.filter_num * self.out_num),
            nn.BatchNorm1d(self.filter_size * self.filter_num * self.out_num))

        self.image_net = nn.Sequential(
            nn.Linear(self.motion_dim, self.image_size * self.filter_num),
            nn.BatchNorm1d(self.image_size * self.filter_num),
            nn.Tanh())

        self.mart_fc = nn.Sequential(
            nn.Linear(self.content_dim + self.cfg.LABEL_NUM, self.content_dim),
            nn.BatchNorm1d(self.content_dim),
            nn.Tanh())

        # ngf x 4 x 4 -> ngf/2 x 8 x 8
        self.upsample1 = upBlock(ngf, ngf // 2)
        # -> ngf/4 x 16 x 16
        self.upsample2 = upBlock(ngf // 2, ngf // 4)

        # For two stage
        if self.cfg.TWO_STG:
            # -> ngf/16 x 32 x 32
            self.upsample3 = upBlock(ngf // 4, ngf // 16)
            self.next_g = NEXT_STAGE_G(ngf // 16, self.cfg.MART.hidden_size, self.content_dim, attn_type=self.cfg.TWO_STG_ATTN)
            self.next_img = nn.Sequential(
            conv3x3(ngf // 16, 3),  # -> 3 x 64 x 64
            nn.Tanh())

        else:
            # -> ngf/8 x 32 x 32
            self.upsample3 = upBlock(ngf // 4, ngf // 8)
            # -> ngf/16 x 64 x 64
            self.upsample4 = upBlock(ngf // 8, ngf // 16)
            # -> 3 x 64 x 64
            self.img = nn.Sequential(
                conv3x3(ngf // 16, 3),
                nn.Tanh())

        # Moda: Adding Seg part
        if self.use_segment:
            ngf_seg = self.gf_dim_seg

            self.seg_c = conv3x3(ngf_seg, ngf)
            self.seg_c1 = conv3x3(ngf_seg//2, ngf//2)
            # self.seg_c2 = conv3x3(ngf_seg//4, ngf//4)
            # self.seg_c3 = conv3x3(ngf_seg//8, ngf//8)
            # self.seg_c4 = conv3x3(ngf_seg//16, ngf//16)

            # For generate seg and img v4 and v5 and v6
            self.fc_seg = nn.Sequential(
                nn.Linear(ninput, ngf_seg * 4 * 4, bias=False),
                nn.BatchNorm1d(ngf_seg * 4 * 4),
                nn.ReLU(True))
            # ngf x 4 x 4 -> ngf/2 x 8 x 8
            self.upsample1_seg = upBlock(ngf_seg, ngf_seg // 2)
            # -> ngf/4 x 16 x 16
            self.upsample2_seg = upBlock(ngf_seg // 2, ngf_seg // 4)
            # -> ngf/8 x 32 x 32
            self.upsample3_seg = upBlock(ngf_seg // 4, ngf_seg // 8)
            # -> ngf/16 x 64 x 64
            self.upsample4_seg = upBlock(ngf_seg // 8, ngf_seg // 16)
            # -> 1 x 64 x 64
            self.img_seg = nn.Sequential(
                conv3x3(ngf_seg // 16, 1),
                nn.Tanh())
            self.presample = nn.Sequential(
                conv3x3(1, ngf_seg // 16),
                nn.BatchNorm2d(ngf_seg // 16),
                nn.ReLU(),
            )
            self.downsample1_seg = downBlock( ngf_seg // 16, ngf_seg // 8 )
            self.downsample2_seg = downBlock( ngf_seg // 8, ngf_seg // 4 )
            self.downsample3_seg = downBlock( ngf_seg // 4, ngf_seg // 2 )
            self.downsample4_seg = downBlock( ngf_seg // 2, ngf_seg )

        self.m_net = nn.Sequential(
            nn.Linear(self.motion_dim, self.motion_dim),
            nn.BatchNorm1d(self.motion_dim))

        self.c_net = nn.Sequential(
            nn.Linear(self.content_dim, self.cfg.MART.hidden_size),
            nn.BatchNorm1d(self.cfg.MART.hidden_size))

        self.dfn_layer = DynamicFilterLayer(self.filter_size,
                                            pad=self.filter_size // 2)

    def sample_videos(self, motion_input, content_input, caption_input_ids=None, caption_input_masks=None, labels=None, seg=False): # Moda: Seg flag is added, also extra return
        content_input = content_input.view(-1, self.cfg.VIDEO_LEN * content_input.shape[2])
        r_code, r_mu, r_logvar = self.ca_net(torch.squeeze(content_input))
        c_mu = r_mu.repeat(self.video_len, 1).view(-1, r_mu.shape[1])

        crnn_code, word_embs, _ = self.motion_content_rnn(caption_input_ids, caption_input_masks, r_code, labels)
        temp = motion_input.view(-1, motion_input.shape[2])
        m_code, m_mu, m_logvar = temp, temp, temp  # self.ca_net(temp)
        m_code = m_code.view(motion_input.shape[0], self.video_len, self.motion_dim)
        zm_code = self.sample_z_motion(m_code, self.video_len)

        # one
        zmc_code = torch.cat((zm_code, c_mu), dim=1)
        # two
        m_image = self.image_net(m_code.view(-1, m_code.shape[2]))

        m_image = m_image.view(-1, self.filter_num, self.image_size)
        c_filter = self.filter_net(crnn_code)
        c_filter = c_filter.view(-1, self.out_num, self.filter_num, self.filter_size)
        mc_image = self.dfn_layer([m_image, c_filter])

        # Moda: next commented code is DuCo code, it is replaced with CP-CSV  (NoCascade) c_code
        # bcs it contains it in the else branch. But becarful, this function has extra return now
        # # INIT_G_START
        # zmc_all = torch.cat((zmc_code, mc_image.squeeze(1)), dim=1)
        # zmc_all = self.fc(zmc_all)
        # zmc_all = zmc_all.view(-1, self.gf_dim, 4, 4)
        # h_code = self.upsample1(zmc_all)
        # h_code = self.upsample2(h_code)
        # h_code = self.upsample3(h_code)
        #
        # ################# NEW #############################
        # # need to fix the dimensions of word_embs
        # # r_code = [b * GAN.CONDITION_DIM]
        # if self.cfg.TWO_STG:
        #     r_code = r_code.unsqueeze(1).repeat(1, self.video_len, 1).view(-1, r_code.shape[-1])
        #     # print(h_code.shape, r_code.shape, word_embs[0].shape)
        #     # h_code2, _ = self.next_g(h_code, r_code, torch.cat(word_embs, dim=0).transpose(-2, -1),
        #     #                          caption_input_masks.view(-1, caption_input_masks.shape[-1]))
        #     seq_len, word_emb_dim = word_embs[0].shape[-2], word_embs[0].shape[-1]
        #     # print(seq_len, word_emb_dim)
        #     # print(torch.stack(word_embs).shape)
        #     # print(torch.stack(word_embs).transpose(1,0).shape)
        #     h_code2, _ = self.next_g(h_code, r_code, torch.stack(word_embs).transpose(1, 0).reshape(-1, seq_len, word_emb_dim).transpose(-2, -1),
        #                              caption_input_masks.view(-1, caption_input_masks.shape[-1]))
        #     h = self.next_img(h_code2)
        # else:
        #     h_code = self.upsample4(h_code)
        #     # INIT_G_END
        #     # state size 3 x 64 x 64
        #     # GET_IMG_G_START
        #     h = self.img(h_code)
        #     # GET_IMG_G_END
        # ################ NEW ###############################
        #
        # fake_video = h.view(int(h.size(0) / self.video_len), self.video_len, self.n_channels, h.size(3), h.size(3))
        # fake_video = fake_video.permute(0, 2, 1, 3, 4)
        # # print("Generated video shape", fake_video.shape)
        # return None, fake_video, m_mu, m_logvar, r_mu, r_logvar


        # Moda: here comes CP-CSV  (NoCascade) code (the replacement). It is with Extra return wrt DuCo original code
        # DuCo MART implementation is added in both brnches of use_segment
        # Basically, it replaces the upsample4 part
        zmc_all_ = torch.cat((zmc_code, mc_image.squeeze(1)), dim = 1)
        zmc_img = self.fc(zmc_all_).view(-1, self.gf_dim, 4, 4)

        if self.use_segment:
			# Moda: Cascade branch
            zmc_seg = self.fc_seg(zmc_all_).view(-1, self.gf_dim_seg, 4, 4)
            # print(zmc_seg.shape)
            h_seg1 = self.upsample1_seg(zmc_seg) #;print(h_seg1.shape)
            h_seg2 = self.upsample2_seg(h_seg1) #;print(h_seg2.shape)
            h_seg3 = self.upsample3_seg(h_seg2) #;print(h_seg3.shape)

            # if self.cfg.TWO_STG: # Moda: Segmentation MART branch (Commented for now)
            #     ## TODO: make sure next_g & next_img are applicable for segmentation also
            #     r_code = r_code.unsqueeze(1).repeat(1, self.video_len, 1).view(-1, r_code.shape[-1])
            #     seq_len, word_emb_dim = word_embs[0].shape[-2], word_embs[0].shape[-1]
            #     h_seg4, _ = self.next_g(h_seg3, r_code, torch.stack(word_embs).transpose(1, 0).reshape(-1, seq_len, word_emb_dim).transpose(-2, -1),
            #                             caption_input_masks.view(-1, caption_input_masks.shape[-1]))
            #     segm_video = self.next_img(h_seg4)
            # else:
            #     h_seg4 = self.upsample4_seg(h_seg3) #;print(h_seg4.shape)
            #     segm_video = self.img_seg(h_seg4)
            h_seg4 = self.upsample4_seg(h_seg3) # Moda: upper cond is commented!
            segm_video = self.img_seg(h_seg4)

			# Moda: cont Cascade
            zmc_latent = self.presample(segm_video) #;print(zmc_latent.shape)
            g_seg4 = self.downsample1_seg(zmc_latent) #;print(g_seg4.shape)
            g_seg3 = self.downsample2_seg(g_seg4) #;print(g_seg3.shape)
            g_seg2 = self.downsample3_seg(g_seg3) #;print(g_seg2.shape)
            g_seg1 = self.downsample4_seg(g_seg2) #;print(g_seg1.shape)

            zmc_img = self.seg_c(g_seg1) * zmc_img + zmc_img

            h_img = self.upsample1(zmc_img)
            h_img = self.seg_c1(g_seg2) * h_img + h_img # batch_size*video_len, 1024, 8, 8

            h_img = self.upsample2(h_img)
            # h_img = self.seg_c2(g_seg3) * h_img + h_img

            h_img = self.upsample3(h_img)
            # h_img = self.seg_c3(g_seg4) * h_img + h_img

            if self.cfg.TWO_STG: # Moda: Image MART branch
                # Moda: r_code, seq_len, word_emb_dim are already set from the prev "TWO_STG" branch
                ## TODO: make sure next_g & next_img are applicable for segmentation also
                r_code = r_code.unsqueeze(1).repeat(1, self.video_len, 1).view(-1, r_code.shape[-1])
                seq_len, word_emb_dim = word_embs[0].shape[-2], word_embs[0].shape[-1]
                h_img2, _ = self.next_g(h_img, r_code, torch.stack(word_embs).transpose(1, 0).reshape(-1, seq_len, word_emb_dim).transpose(-2, -1),
                                        caption_input_masks.view(-1, caption_input_masks.shape[-1]))
                fake_video = self.next_img(h_img2)
            else:
                h_img = self.upsample4(h_img) #;print(h_img.shape)
                fake_video = self.img_seg(h_img)

            segm_temp = segm_video.view(-1, self.video_len, 1, self.segment_size, self.segment_size)
            segm_temp = segm_temp.permute(0, 2, 1, 3, 4)
            # generate video
            # Moda: The next line is commented, bcs it's replaced by the prev "TWO_STG" condition (MART)
            # fake_video = self.img(h_img) # Moda
            fake_video = fake_video.view(-1, self.video_len, self.n_channels, self.segment_size, self.segment_size)
            fake_video = fake_video.permute(0, 2, 1, 3, 4)

            if seg==True:
                return ((zmc_seg, h_seg1, h_seg2, h_seg3), (g_seg1, g_seg2, g_seg3, g_seg4)), \
                    fake_video, m_mu, m_logvar, r_mu, r_logvar, segm_video # m_mu(60,365), m_logvar(60,365), r_mu(12,124), r_logvar(12,124)
            else:
                return ((zmc_seg, h_seg1, h_seg2, h_seg3), (g_seg1, g_seg2, g_seg3, g_seg4)), \
                    fake_video, m_mu, m_logvar, r_mu, r_logvar, None # m_mu(60,365), m_logvar(60,365), r_mu(12,124), r_logvar(12,124)
        else:
            h_code = self.upsample1(zmc_img) # h_code: batch_size*video_len, 1024, 8, 8 *
            h_code = self.upsample2(h_code)  # h_code: batch_size*video_len, 512, 16, 16 *
            h_code = self.upsample3(h_code)  # h_code: batch_size*video_len, 256, 32, 32 *

            # Moda: MART implememntation
            if self.cfg.TWO_STG:
                r_code = r_code.unsqueeze(1).repeat(1, self.video_len, 1).view(-1, r_code.shape[-1])
                # print(h_code.shape, r_code.shape, word_embs[0].shape)
                # h_code2, _ = self.next_g(h_code, r_code, torch.cat(word_embs, dim=0).transpose(-2, -1),
                #                          caption_input_masks.view(-1, caption_input_masks.shape[-1]))
                seq_len, word_emb_dim = word_embs[0].shape[-2], word_embs[0].shape[-1]
                # print(seq_len, word_emb_dim)
                # print(torch.stack(word_embs).shape)
                # print(torch.stack(word_embs).transpose(1,0).shape)
                h_code2, _ = self.next_g(h_code, r_code, torch.stack(word_embs).transpose(1, 0).reshape(-1, seq_len, word_emb_dim).transpose(-2, -1),
                                         caption_input_masks.view(-1, caption_input_masks.shape[-1]))
                h = self.next_img(h_code2)
            else:
                # Moda: the last step of h_code is inside the else
                h_code = self.upsample4(h_code)  # h_code: batch_size*video_len=60, 128, 64, 64 *
                # state size 3 x 64 x 64
                h = self.img(h_code) ## *
            fake_video = h.view( int(h.size(0)/self.video_len), self.video_len, self.n_channels, h.size(3), h.size(3)) # 12, 5, 3, 64, 64
            fake_video = fake_video.permute(0, 2, 1, 3, 4) # 12, 3, 5, 64, 64
            #pdb.set_trace()
            return None, fake_video, m_mu, m_logvar, r_mu, r_logvar, None # m_mu(60,365), m_logvar(60,365), r_mu(12,124), r_logvar(12,124)

    def sample_images(self, motion_input, content_input, caption_input_ids=None, caption_input_masks=None, labels=None, seg=False): # Moda: Seg flag is added, also extra return
        m_code, m_mu, m_logvar = motion_input, motion_input, motion_input  # self.ca_net(motion_input)
        # print(content_input.shape, cfg.VIDEO_LEN)
        # content_input = content_input.view(-1, cfg.VIDEO_LEN * content_input.shape[2])
        content_input = torch.reshape(content_input, (-1, self.cfg.VIDEO_LEN * content_input.shape[2]))
        r_code, c_mu, c_logvar = self.ca_net(content_input)
        crnn_code, word_embs, _ = self.motion_content_rnn(caption_input_ids.unsqueeze(1), caption_input_masks.unsqueeze(1), c_mu, labels.unsqueeze(1)) # wut

        zm_code = self.sample_z_motion(m_code, 1)
        # one
        zmc_code = torch.cat((zm_code, c_mu), dim=1)
        # two
        m_image = self.image_net(m_code)
        m_image = m_image.view(-1, self.filter_num, self.image_size)
        c_filter = self.filter_net(crnn_code)
        c_filter = c_filter.view(-1, self.out_num, self.filter_num, self.filter_size)

        mc_image = self.dfn_layer([m_image, c_filter])


        # Moda: next commented code is DuCo code, it is replaced with CP-CSV  (NoCascade) c_code
        # bcs it contains it in the else branch. But becarful, this function has extra return now
        # zmc_all = torch.cat((zmc_code, mc_image.squeeze(1)), dim=1)
        # # combine
        # zmc_all = self.fc(zmc_all)
        # zmc_all = zmc_all.view(-1, self.gf_dim, 4, 4)
        # h_code = self.upsample1(zmc_all)
        # h_code = self.upsample2(h_code)
        # h_code = self.upsample3(h_code)
        #
        # if self.cfg.TWO_STG:
        #     h_code2, _ = self.next_g(h_code, c_code, word_embs[0].transpose(-2, -1), caption_input_masks)
        #     fake_img = self.next_img(h_code2)
        # else:
        #     h_code = self.upsample4(h_code)
        #     # state size 3 x 64 x 64
        #     fake_img = self.img(h_code)
        #
        # # print("Generated image shape", fake_img.shape)
        #
        # return None, fake_img, m_mu, m_logvar, c_mu, c_logvar

        # Moda: here comes CP-CSV  (NoCascade) code (the replacement). It is with Extra return wrt DuCo original code
        # DuCo MART implementation is added in both brnches of use_segment
        # Basically, it replaces the upsample4 part
        zmc_all_ = torch.cat((zmc_code, mc_image.squeeze(1)), dim = 1)
        zmc_img = self.fc(zmc_all_).view(-1, self.gf_dim, 4, 4)

        if self.use_segment:
			# Moda: Cascade branch
            zmc_seg = self.fc_seg(zmc_all_).view(-1, self.gf_dim_seg, 4, 4)
            # print(zmc_seg.shape)
            h_seg1 = self.upsample1_seg(zmc_seg) #;print(h_seg1.shape)
            h_seg2 = self.upsample2_seg(h_seg1) #;print(h_seg2.shape)
            h_seg3 = self.upsample3_seg(h_seg2) #;print(h_seg3.shape)

            # if self.cfg.TWO_STG: # Moda: Segmentation MART branch (Commented for now)
            #     ## TODO: make sure next_g & next_img are applicable for segmentation also
            #     r_code = r_code.unsqueeze(1).repeat(1, self.video_len, 1).view(-1, r_code.shape[-1])
            #     seq_len, word_emb_dim = word_embs[0].shape[-2], word_embs[0].shape[-1]
            #     h_seg4, _ = self.next_g(h_seg3, r_code, torch.stack(word_embs).transpose(1, 0).reshape(-1, seq_len, word_emb_dim).transpose(-2, -1),
            #                                 caption_input_masks.view(-1, caption_input_masks.shape[-1]))
            #     segm_img = self.next_img(h_seg4)
            # else:
            #     h_seg4 = self.upsample4_seg(h_seg3) #;print(h_seg4.shape)
            #     segm_img = self.img_seg(h_seg4)
            h_seg4 = self.upsample4_seg(h_seg3) # Moda: upper cond is commented!
            segm_img = self.img_seg(h_seg4)

			# Moda: cont Cascade
            zmc_latent = self.presample(segm_img) #;print(zmc_latent.shape)
            g_seg4 = self.downsample1_seg(zmc_latent) #;print(g_seg4.shape)
            g_seg3 = self.downsample2_seg(g_seg4) #;print(g_seg3.shape)
            g_seg2 = self.downsample3_seg(g_seg3) #;print(g_seg2.shape)
            g_seg1 = self.downsample4_seg(g_seg2) #;print(g_seg1.shape)

            zmc_img = self.seg_c(g_seg1) * zmc_img + zmc_img

            h_img = self.upsample1(zmc_img)
            h_img = self.seg_c1(g_seg2) * h_img + h_img # batch_size*video_len, 1024, 8, 8

            h_img = self.upsample2(h_img)
            # h_img = self.seg_c2(g_seg3) * h_img + h_img

            h_img = self.upsample3(h_img)
            # h_img = self.seg_c3(g_seg4) * h_img + h_img

            if self.cfg.TWO_STG: # Moda: Image MART branch
                # Moda: r_code, seq_len, word_emb_dim are already set from the prev "TWO_STG" branch
                ## TODO: make sure next_g & next_img are applicable for segmentation also
                # r_code = r_code.unsqueeze(1).repeat(1, self.video_len, 1).view(-1, r_code.shape[-1])
                seq_len, word_emb_dim = word_embs[0].shape[-2], word_embs[0].shape[-1]
                h_img2, _ = self.next_g(h_img, r_code, torch.stack(word_embs).transpose(1, 0).reshape(-1, seq_len, word_emb_dim).transpose(-2, -1),
                                            caption_input_masks.view(-1, caption_input_masks.shape[-1]))
                fake_img = self.next_img(h_img2)
            else:
                h_img = self.upsample4(h_img) #;print(h_img.shape)
                fake_img = self.img(h_img)

            if seg==True:
                return ((zmc_seg, h_seg1, h_seg2, h_seg3), (g_seg1, g_seg2, g_seg3, g_seg4)), \
                    fake_img, m_mu, m_logvar, c_mu, c_logvar, segm_img
            else:
                return ((zmc_seg, h_seg1, h_seg2, h_seg3), (g_seg1, g_seg2, g_seg3, g_seg4)), \
                         fake_img, m_mu, m_logvar, c_mu, c_logvar, None
        else:
            h_code = self.upsample1(zmc_img) # h_code: batch_size*video_len, 1024, 8, 8 *
            h_code = self.upsample2(h_code)  # h_code: batch_size*video_len, 512, 16, 16 *
            h_code = self.upsample3(h_code)  # h_code: batch_size*video_len, 256, 32, 32 *

            # Moda: MART implememntation
            if self.cfg.TWO_STG:
                r_code = r_code.unsqueeze(1).repeat(1, self.video_len, 1).view(-1, r_code.shape[-1])
                # print(h_code.shape, r_code.shape, word_embs[0].shape)
                # h_code2, _ = self.next_g(h_code, r_code, torch.cat(word_embs, dim=0).transpose(-2, -1),
                #                          caption_input_masks.view(-1, caption_input_masks.shape[-1]))
                seq_len, word_emb_dim = word_embs[0].shape[-2], word_embs[0].shape[-1]
                # print(seq_len, word_emb_dim)
                # print(torch.stack(word_embs).shape)
                # print(torch.stack(word_embs).transpose(1,0).shape)
                h_code2, _ = self.next_g(h_code, r_code, torch.stack(word_embs).transpose(1, 0).reshape(-1, seq_len, word_emb_dim).transpose(-2, -1),
                                         caption_input_masks.view(-1, caption_input_masks.shape[-1]))
                fake_img = self.next_img(h_code2)
            else:
                # Moda: the last step of h_code is inside the else
                h_code = self.upsample4(h_code)  # h_code: batch_size*video_len=60, 128, 64, 64 *
                # state size 3 x 64 x 64
                fake_img = self.img(h_code) ## *
            return None, fake_img, m_mu, m_logvar, c_mu, c_logvar, None

    def train_autoencoder(self, real_segments):
        zmc_latent = self.presample(real_segments) #;print(zmc_latent.shape)
        g_seg4 = self.downsample1_seg(zmc_latent) #;print(g_seg4.shape)
        g_seg3 = self.downsample2_seg(g_seg4) #;print(g_seg3.shape)
        g_seg2 = self.downsample3_seg(g_seg3) #;print(g_seg2.shape)
        g_seg1 = self.downsample4_seg(g_seg2) #;print(g_seg1.shape)

        h_seg1 = self.upsample1_seg(g_seg1) #;print(h_seg1.shape)
        h_seg2 = self.upsample2_seg(h_seg1) #;print(h_seg2.shape)
        h_seg3 = self.upsample3_seg(h_seg2) #;print(h_seg3.shape)
        h_seg4 = self.upsample4_seg(h_seg3) #;print(h_seg4.shape)
        segm_img = self.img_seg(h_seg4)
        return segm_img #(( h_seg1, h_seg2, h_seg3, h_seg4), (g_seg1, g_seg2, g_seg3, g_seg4))
    
    def motion_content_rnn(self, word_input_ids_list, input_masks_list, c_code, labels, return_memory=False): # Moda: over write NoMART method
        """
        Args:
            input_ids_list: [(N, L)] * step_size
            input_masks_list: [(N, L)] * step_size with 1 indicates valid bits
                will not be used when return_memory is True, thus can be None in this case
            return_memory: bool,
        Returns:
        """
        # [(N, M, D)] * num_hidden_layers, initialized internally

        c_code = self.c_net(c_code)

        memory_list = []
        memory_list.append([c_code.unsqueeze(1).repeat(1, self.cfg.MART.n_memory_cells, 1), c_code.unsqueeze(1).repeat(1, self.cfg.MART.n_memory_cells, 1)])
        video_len = word_input_ids_list.shape[1]

        encoded_outputs_list = []  # [(N, L, D)] * step_size
        for idx in range(video_len):
            ms, encoded_layer_outputs = self.mart_forward_step(memory_list[-1],
                                                               word_input_ids_list[:, idx, :],
                                                               input_masks_list[:, idx, :])

            memory_list.append(ms)
            encoded_outputs_list.append(encoded_layer_outputs[-1])

        c_m_t = [self.pooler(h_k, input_masks_list[:, idx, :]).unsqueeze(1) for idx, h_k in enumerate(encoded_outputs_list)]
        mocornn_co = self.mart_fc(torch.cat((torch.cat(c_m_t, dim=1), labels), dim=-1).view(-1, self.content_dim + self.cfg.LABEL_NUM))

        if return_memory:  # used to analyze memory
            return mocornn_co, encoded_outputs_list, memory_list
        else:  # normal training/evaluation mode
            return mocornn_co, encoded_outputs_list, None

class STAGE1_D_IMG(nn.Module):
    def __init__(self, cfg, use_categories = True):
        super(STAGE1_D_IMG, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.text_dim = cfg.TEXT.DIMENSION
        self.label_num = cfg.LABEL_NUM
        self.cfg = cfg
        self.final_stride = 2
        self.define_module(use_categories)

    def define_module(self, use_categories):
        ndf, nef = self.df_dim, self.ef_dim
        self.encode_img = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), # Moda: in CP-CSV (NoCascade), this is passed to "spectral_norm()"
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*2) x 16 x 16
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False), # Moda: in CP-CSV (NoCascade), this is passed to "spectral_norm()"
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*4) x 8 x 8
            nn.Conv2d(ndf*4, ndf * 8, 4, self.final_stride, 1, bias=False), # Moda: in CP-CSV (NoCascade), final_stride is not available but hardcoded
            nn.BatchNorm2d(ndf * 8),
            # state size (ndf * 8) x 4 x 4)
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.seq_consisten_model = None # Mode: this line is available in CP-CSV (NoCascade)
        self.get_cond_logits = D_GET_LOGITS(ndf, nef + self.text_dim + self.label_num)
        self.get_uncond_logits = None

        if use_categories:
            self.cate_classify = nn.Conv2d(ndf * 8, self.label_num, 4, 4, 1, bias = False)
        else:
            self.cate_classify = None

        if self.cfg.IMG_DUAL: # This branch is just for DuCo
            self.embed = nn.Linear(2048, self.cfg.EMBED_SIZE)
            self.decoder = DecoderRNN(embed_size=self.cfg.EMBED_SIZE,
                                      hidden_size=self.cfg.HIDDEN_SIZE,
                                      vocab_size=self.cfg.VOCAB_SIZE,
                                      pretrained_embeddings = self.cfg.MART.vocab_glove_path)

    def forward(self, image):
        img_embedding = self.encode_img(image)
        #(60,3,64,64) -> (60,992,4,4)
        # print(img_embedding.shape)
        return img_embedding

    def get_captions(self, features, captions=None): # Moda: This branch is just for DuCo
        batch_size = features.size(0)
        features = torch.mean(features, dim=-1).view(batch_size, -1)
        embeddings = nn.functional.relu(self.embed(features))
        outputs = self.decoder(embeddings, captions)
        return outputs.view(-1, self.decoder.vocab_size)

class STAGE1_D_SEG(nn.Module): # Moda: extra class for segmentation
    def __init__(self, cfg, use_categories = True):
        super(STAGE1_D_SEG, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.text_dim = cfg.TEXT.DIMENSION
        self.label_num = cfg.LABEL_NUM
        self.cfg = cfg
        self.final_stride = 2
        self.define_module(use_categories)

    def define_module(self, use_categories):
        ndf, nef = self.df_dim, self.ef_dim
        self.encode_img = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False), # Moda: 1st parameter is "1" instead of "3" (segmentation = 1-channel)
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), # Moda: in CP-CSV (NoCascade), this is passed to "spectral_norm()"
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*2) x 16 x 16
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False), # Moda: in CP-CSV (NoCascade), this is passed to "spectral_norm()"
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*4) x 8 x 8
            nn.Conv2d(ndf*4, ndf * 8, 4, self.final_stride, 1, bias=False), # Moda: in CP-CSV (NoCascade), final_stride is not available but hardcoded
            nn.BatchNorm2d(ndf * 8),
            # state size (ndf * 8) x 4 x 4)
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.seq_consisten_model = None # Mode: this line is available in CP-CSV (NoCascade)
        self.get_cond_logits = D_GET_LOGITS(int(ndf), nef + self.text_dim + self.label_num) # Moda: first item is casted to "int()"
        self.get_uncond_logits = None

        if use_categories:
            self.cate_classify = nn.Conv2d(ndf * 8, self.label_num, 4, 4, 1, bias = False)
        else:
            self.cate_classify = None

        if self.cfg.IMG_DUAL: # Moda: This branch is just for DuCo
            self.embed = nn.Linear(2048, self.cfg.EMBED_SIZE) # Moda: 2048 might need to be checked later
            self.decoder = DecoderRNN(embed_size=self.cfg.EMBED_SIZE,
                                      hidden_size=self.cfg.HIDDEN_SIZE,
                                      vocab_size=self.cfg.VOCAB_SIZE,
                                      pretrained_embeddings = self.cfg.MART.vocab_glove_path)

    def forward(self, image):
        img_embedding = self.encode_img(image)
        #(60,3,64,64) -> (60,992,4,4)
        # print(img_embedding.shape)
        return img_embedding

    def get_captions(self, features, captions=None): # This function is just for DuCo
        batch_size = features.size(0)
        features = torch.mean(features, dim=-1).view(batch_size, -1)
        embeddings = nn.functional.relu(self.embed(features))
        outputs = self.decoder(embeddings, captions)
        return outputs.view(-1, self.decoder.vocab_size)

class DecoderRNN(nn.Module): # Moda: this is just for DuCo repo, N/A in CP-CSV (NoCascade)

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, pretrained_embeddings = ''):
        super().__init__()
        if pretrained_embeddings != '':
            embeds = torch.tensor(torch.load(pretrained_embeddings))
            self.embedding_layer = nn.Embedding.from_pretrained(embeds)
        else:
            self.embedding_layer = nn.Embedding(vocab_size, embed_size)

        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.linear = nn.Linear(hidden_size, vocab_size)

        self.vocab_size = vocab_size

    def forward(self, features, captions):
        captions = captions[:, :-1]
        embed = self.embedding_layer(captions)
        embed = torch.cat((features.unsqueeze(1), embed), dim=1)
        lstm_outputs, _ = self.lstm(embed)
        out = self.linear(lstm_outputs)

        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        output_sentence = []
        for i in range(max_len):
            lstm_outputs, states = self.lstm(inputs, states)
            lstm_outputs = lstm_outputs.squeeze(1)
            out = self.linear(lstm_outputs)
            last_pick = out.max(1)[1]
            output_sentence.append(last_pick.item())
            inputs = self.embedding_layer(last_pick).unsqueeze(1)

        return output_sentence

class STAGE1_D_STY_V2(nn.Module):
    def __init__(self, cfg):
        super(STAGE1_D_STY_V2, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.text_dim = cfg.TEXT.DIMENSION
        self.label_num = cfg.LABEL_NUM
        self.cfg = cfg
        self.define_module()

    # TODO: This module does not align with that paper says?? Is there a better way to do story discrimination
    def define_module(self):
        ndf, nef = self.df_dim, self.ef_dim
        self.encode_img = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False), # Moda: in CP-CSV (NoCascade), this is passed to "spectral_norm()"
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), # Moda: in CP-CSV (NoCascade), this is passed to "spectral_norm()"
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*2) x 16 x 16
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False), # Moda: in CP-CSV (NoCascade), this is passed to "spectral_norm()"
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*4) x 8 x 8
            nn.Conv2d(ndf*4, ndf * 8, 4, 2, 1, bias=False), # Moda: in CP-CSV (NoCascade), this is passed to "spectral_norm()"
            nn.BatchNorm2d(ndf * 8),
            # state size (ndf * 8) x 4 x 4)
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Moda: this branch is available in CP-CSV (NoCascade)
        self.seq_consisten_model = None
        if cfg.USE_SEQ_CONSISTENCY:
            self.seq_consisten_model = VideoEncoder()
        #     # checkpoint = torch.load('logs/consistencybaseline_0.5/model.pt')
        #     # self.seq_consisten_model.load_state_dict(checkpoint['model'].state_dict())

        self.get_cond_logits = D_GET_LOGITS(ndf, nef + self.text_dim + self.label_num)
        self.get_uncond_logits = None
        self.cate_classify = None

    def forward(self, story):
        N, C, video_len, W, H = story.shape
        story = story.permute(0,2,1,3,4)
        story = story.contiguous().view(-1, C,W,H)
        story_embedding = torch.squeeze(self.encode_img(story))
        _, C1, W1, H1 = story_embedding.shape
        story_embedding = story_embedding.view(N,video_len, C1, W1, H1)
        # story_embedding = story_embedding.mean(1).squeeze() # Moda: this line is not commented in CP-CSV (NoCascade)
        return story_embedding

    def extract_img_features(self, input_images_list, total_seq_len, device): # This function is just for DuCo
        input_imgs = torch.cat(input_images_list, dim=0).to(device)
        bsz = input_images_list[0].shape[0]
        # print(input_imgs.shape)
        features = self.feature_extractor(input_imgs).permute(0, 2, 3, 1).view(-1, 64, 2048)
        # print(features.shape)
        outputs = [torch.zeros(bsz, total_seq_len, 2048).to(device) for _ in range(len(input_images_list))]
        for i in range(len(input_images_list)):
            outputs[i][:, 1:65, :] = features[i * bsz:(i + 1) * bsz, :, :]
        return outputs
