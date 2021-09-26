from __future__ import print_function
from six.moves import range

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import numpy as np
import pickle
from tqdm import tqdm

from .miscc.config import cfg
from .miscc.utils import mkdir_p
from .miscc.utils import weights_init
from .miscc.utils import save_story_results, save_model, save_test_samples
from .miscc.utils import KL_loss
from .miscc.utils import compute_discriminator_loss, compute_generator_loss, compute_dual_captioning_loss
from shutil import copyfile
from torchvision.models import vgg16

# Imports for dual learning with video captioning model
from train_mart import init_feature_extractor
from mart.recurrent import RecursiveTransformer
import torchvision.transforms as transforms

# Moda: import model classes and new classes
from .model import CreateModel, StoryGAN, STAGE1_D_IMG, STAGE1_D_SEG, STAGE1_D_STY_V2, STAGE1_D_SEG
from .model import NoMartNoCascade, MartNoCascade, NoMartCascade, MartCascade


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self, out_images, target_images):
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        return perception_loss

class GANTrainer(object):
    def __init__(self, cfg, output_dir=None, ratio = 1.0):
        if cfg.TRAIN.FLAG:
            assert output_dir, "Output directory is required for training"
            output_dir = output_dir + '_r' + str(ratio) + '/'
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'Log')
            self.test_dir = os.path.join(output_dir, 'Test')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
            mkdir_p(self.test_dir)
            # Moda: the new "model.py" contains all types of models
            copyfile('./dcsgan/model.py', os.path.join(output_dir, 'model.py'))

        self.video_len = cfg.VIDEO_LEN
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        assert cfg.IMG_DISC or cfg.STORY_DISC
        self.use_image_disc = cfg.IMG_DISC
        self.use_story_disc = cfg.STORY_DISC
        self.use_mart = cfg.USE_MART
        self.use_segment = cfg.SEGMENT_LEARNING # Moda

        # self.gpus = []
        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        self.num_gpus = len(self.gpus)
        self.imbatch_size = cfg.TRAIN.IM_BATCH_SIZE * self.num_gpus
        self.stbatch_size = cfg.TRAIN.ST_BATCH_SIZE * self.num_gpus
        self.ratio = ratio

        self.cfg = cfg
        
        # Moda
        ## TODO: Add continue_ckpt if needed
        # self.con_ckpt = args.continue_ckpt

        if cfg.IMG_DUAL or cfg.STORY_DUAL:
            assert cfg.VOCAB_SIZE is not None
        self.img_dual = cfg.IMG_DUAL
        self.story_dual = cfg.STORY_DUAL
        
        self.cuda_is_available = False # Moda-fix: add condition
        self.map_location = 'cpu' # Moda-fix: add map_location, might be redundant!
        if torch.cuda.is_available():
            torch.cuda.set_device(self.gpus[0])
            cudnn.benchmark = True
            self.cuda_is_available = True
            self.map_location=lambda storage, loc: storage.cuda() # Moda-fix: might be redundant!

        if cfg.TRAIN.PERCEPTUAL_LOSS:
            self.perceptual_loss_net = PerceptualLoss()

    # ############# For training stageI GAN #############
    def load_network_stageI(self): # Moda: extra paramters are added
        # Moda: no need for this condition and the import is replaced
        # from .model import StoryGAN, STAGE1_D_IMG, STAGE1_D_STY_V2, StoryMartGAN
        # if cfg.CASCADE_MODEL:
        #     from cascade_model import StoryGAN, STAGE1_D_IMG, STAGE1_D_STY_V2, STAGE1_D_SEG, StoryMartGAN
        # else:
        #     from model import CreateModel, StoryGAN, STAGE1_D_IMG, STAGE1_D_STY_V2, STAGE1_D_SEG, StoryMartGAN

        # Moda: define the training parameters

        # Moda: model creation is replaced by the new function and flags
        # if self.use_mart:
        #     netG = StoryMartGAN(self.cfg, self.video_len)
        # else:
        #     netG = StoryGAN(self.cfg, self.video_len)
        netG = CreateModel(self.use_mart, self.use_segment, self.cfg, self.video_len)

        netG.apply(weights_init)
        print(netG)

        if self.use_image_disc:
            if self.cfg.DATASET_NAME == 'youcook2':
                use_categories = False
            else:
                use_categories = True

            netD_im = STAGE1_D_IMG(self.cfg, use_categories=use_categories)
            netD_im.apply(weights_init)
            print(netD_im)

            if self.cfg.NET_D != '':
                state_dict = \
                    torch.load(self.cfg.NET_D,
                               map_location=lambda storage, loc: storage)
                netD_im.load_state_dict(state_dict)
                print('Load from: ', self.cfg.NET_D)
        else:
            netD_im = None

        if self.use_story_disc:
            netD_st = STAGE1_D_STY_V2(self.cfg)
            netD_st.apply(weights_init)
            print(netD_st)
        else:
            netD_st = None

        # Moda
        netD_se = None
        if self.use_segment:
            netD_se = STAGE1_D_SEG(self.cfg) # v2 # Moda-fix: add argument
            netD_se.apply(weights_init)
            print(netD_se)

        if self.cfg.NET_G != '':
            state_dict = \
                torch.load(self.cfg.NET_G,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load from: ', self.cfg.NET_G)

        if self.cfg.CUDA and self.cuda_is_available: # Moda-fix: add cuda condition
            netG.cuda()
            if self.use_image_disc:
                netD_im.cuda()
            if self.use_story_disc:
                netD_st.cuda()
            # Moda
            if self.use_segment:
                netD_se.cuda()

            if self.cfg.TRAIN.PERCEPTUAL_LOSS:
                self.perceptual_loss_net.loss_network.cuda()

        total_params = sum(p.numel() for p in netD_st.parameters() if p.requires_grad) + sum(
            p.numel() for p in netD_im.parameters() if p.requires_grad) + sum(
            p.numel() for p in netG.parameters() if p.requires_grad) + sum( # Moda
            p.numel() for p in netD_se.parameters() if p.requires_grad)
        print("Total Parameters: %s", total_params)

        # Moda
        ## TODO: Add continue_ckpt if needed
        # if self.con_ckpt:
        #     print('Continue training from epoch {}'.format(self.con_ckpt))
        #     path = '{}/netG_epoch_{}.pth'.format(self.model_dir, self.con_ckpt)
        #     netG.load_state_dict(torch.load(path))
        #     path = '{}/netD_im_epoch_last.pth'.format(self.model_dir)
        #     netD_im.load_state_dict(torch.load(path))
        #     path = '{}/netD_st_epoch_last.pth'.format(self.model_dir)
        #     netD_st.load_state_dict(torch.load(path))
        #     if self.use_segment:
        #         path = '{}/netD_se_epoch_last.pth'.format(self.model_dir)
        #         netD_se.load_state_dict(torch.load(path))

        # Moda
        ## TODO: Fix return, be careful with the extra return::: netD_se
        return netG, netD_im, netD_st, netD_se

    def load_dual_model(self):

        if self.cfg.STORY_DUAL:

            transform = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            checkpoint = torch.load(os.path.join(self.cfg.MART.CKPT_PATH),\
                                                map_location=lambda storage, loc: storage) # Moda-fix: add map_location
            model_config = checkpoint["model_cfg"]
            model = RecursiveTransformer(model_config)
            model.load_state_dict(checkpoint["model"])
            model.max_v_len = model_config.max_v_len
            model.max_t_len = model.config.max_t_len
            feature_extractor = init_feature_extractor()
            model.eval()

            for p in model.parameters():
                p.requires_grad = False

            if self.cfg.CUDA and self.cuda_is_available: # Moda-fix: add cuda condition
                model.cuda()
            return model, feature_extractor, transform
        else:
            raise ValueError

    def sample_real_image_batch(self):
        if self.imagedataset is None:
            self.imagedataset = enumerate(self.imageloader)
        batch_idx, batch = next(self.imagedataset)
        b = batch
        if self.cfg.CUDA and self.cuda_is_available: # Moda-fix: add cuda condition
            for k, v in batch.items():
                if k == 'text':
                    continue
                else:
                    b[k] = v.cuda()

        if batch_idx == len(self.imageloader) - 1:
            self.imagedataset = enumerate(self.imageloader)
        return b

    # Moda
    ## TODO: Check if calculate_vfid and calculate_ssim are needed
    ## Update: mostly not!!!
    # def calculate_vfid(self, netG, epoch, testloader):
    #     netG.eval()
    #     with torch.no_grad():
    #         eval_modeldataset = StoryGANDataset(netG, len(testloader), testloader.dataset)
    #         vfid_value = vfid_score(IgnoreLabelDataset(testloader.dataset),
    #             eval_modeldataset, cuda=True, normalize=True, r_cache='.cache/seg_story_vfid_reference_score.npz'
    #         )
    #         fid_value = fid_score(IgnoreLabelDataset(testloader.dataset),
    #                 eval_modeldataset, cuda=True, normalize=True, r_cache='.cache/seg_story_fid_reference_score.npz'
    #             )
    #     netG.train()
    #
    #     if self._logger:
    #         self._logger.add_scalar('Evaluation/vfid',  vfid_value,  epoch)
    #         self._logger.add_scalar('Evaluation/fid',  fid_value,  epoch)
    #
    # def calculate_ssim(self, netG, epoch, testloader):
    #     netG.eval()
    #     print('calculating SSIM')
    #     with torch.no_grad():
    #         eval_modeldataset = StoryGANSSIMDataset(netG, len(testloader), testloader.dataset)
    #         ssim_value = ssim_score(eval_modeldataset)
    #     netG.train()
    #     print('Epoch: {:d} ssim: {:.4f} ' .format(epoch, ssim_value) )
    #     if self._logger:
    #         self._logger.add_scalar('Evaluation/ssim', ssim_value, epoch)


    def train(self, imageloader, storyloader, testloader, stage=1):
        c_time = time.time()    # Moda
        self.imageloader = imageloader
        self.imagedataset = None
        netG, netD_im, netD_st, netD_se = self.load_network_stageI() # Moda
        start = time.time() # Moda

        if self.cfg.STORY_DUAL:
            netDual, img_ft_extractor, transform = self.load_dual_model()

        # Initial Labels
        im_real_labels = Variable(torch.FloatTensor(self.imbatch_size).fill_(1))
        im_fake_labels = Variable(torch.FloatTensor(self.imbatch_size).fill_(0))
        st_real_labels = Variable(torch.FloatTensor(self.stbatch_size).fill_(1))
        st_fake_labels = Variable(torch.FloatTensor(self.stbatch_size).fill_(0))
        if self.cfg.CUDA and self.cuda_is_available: #Moda-fix: add cuda condition
            im_real_labels, im_fake_labels = im_real_labels.cuda(), im_fake_labels.cuda()
            st_real_labels, st_fake_labels = st_real_labels.cuda(), st_fake_labels.cuda()

        # Moda : Add segmentation weights
        use_segment = cfg.SEGMENT_LEARNING
        segment_weight = cfg.SEGMENT_RATIO
        image_weight = cfg.IMAGE_RATIO

        # Optimizer and Scheduler
        generator_lr = self.cfg.TRAIN.GENERATOR_LR
        discriminator_lr = self.cfg.TRAIN.DISCRIMINATOR_LR
        lr_decay_step = self.cfg.TRAIN.LR_DECAY_EPOCH

        if self.use_image_disc:
            im_optimizerD = \
                optim.Adam(netD_im.parameters(), lr=self.cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.5, 0.999))
        if self.use_story_disc:
            st_optimizerD = \
                optim.Adam(netD_st.parameters(), lr=self.cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.5, 0.999))
        # Moda
        if self.use_segment:
            se_optimizerD = \
                optim.Adam(netD_se.parameters(), lr=self.cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.5, 0.999))

        netG_para = []
        for p in netG.parameters():
            if p.requires_grad:
                netG_para.append(p)
        optimizerG = optim.Adam(netG_para, lr=self.cfg.TRAIN.GENERATOR_LR, betas=(0.5, 0.999))

        # Moda: add mse_loss for video_latents branch
        mse_loss = nn.MSELoss()

        # Moda
        ## TODO:
        # 1- Add if if con_ckp
        # 2- Modify for Epoch loop and add start_epoch
        # 3- Check loss_collector
        # # Start training
        # if not self.con_ckpt:
        #     start_epoch = 0
        # else:
        #     start_epoch = int(self.con_ckpt)
        # print('LR DECAY EPOCH: {}'.format(lr_decay_step))

        loss_collector = []

        count = 0
        # save_test_samples(netG, testloader, self.test_dir, epoch=0, mart=self.use_mart)

        #save_test_samples(netG, testloader, self.test_dir)
        for epoch in range(self.max_epoch + 1):
            # Moda-XXX
            if epoch > 1: break
            print(f">>> epoch ::: {epoch}/{self.max_epoch + 1}")
            l = self.ratio * (2. / (1. + np.exp(-10. * epoch)) - 1)
            start_t = time.time()
            if epoch % lr_decay_step == 0 and epoch > 0:
                generator_lr *= 0.5
                for param_group in optimizerG.param_groups:
                    param_group['lr'] = generator_lr
                discriminator_lr *= 0.5
                if self.use_story_disc:
                    for param_group in st_optimizerD.param_groups:
                        param_group['lr'] = discriminator_lr
                if self.use_image_disc:
                    for param_group in im_optimizerD.param_groups:
                        param_group['lr'] = discriminator_lr
                # Moda
                if self.use_segment:
                    for param_group in se_optimizerD.param_groups:
                        param_group['lr'] = discriminator_lr

            for i, data in tqdm(enumerate(storyloader, 0)):
                # Moda-XXX
                if i > 0: break
                print(f">>> storyloader ::: {i}/{len(storyloader)}")
                ######################################################
                # (1) Prepare training data
                ######################################################
                im_batch = self.sample_real_image_batch()
                st_batch = data

                im_real_cpu = im_batch['images']
                im_motion_input = im_batch['description'][:, :self.cfg.TEXT.DIMENSION] # description vector and arrtibute (60, 356)
                im_content_input = im_batch['content'][:, :, :self.cfg.TEXT.DIMENSION] # description vector and attribute for every story (60,5,356)
                im_real_imgs = Variable(im_real_cpu)
                im_motion_input = Variable(im_motion_input)
                im_content_input = Variable(im_content_input)
                im_labels = Variable(im_batch['labels'])
                if self.use_mart or self.img_dual:
                    im_input_ids = Variable(im_batch['input_id'])
                    im_masks = Variable(im_batch['mask'])

                st_real_cpu = st_batch['images']
                st_motion_input = st_batch['description'][:, :, :self.cfg.TEXT.DIMENSION] #(12,5,356)
                st_content_input = st_batch['description'][:, :, :self.cfg.TEXT.DIMENSION] # (12,5,356)
                st_texts = st_batch['text']
                st_real_imgs = Variable(st_real_cpu)
                st_motion_input = Variable(st_motion_input)
                st_content_input = Variable(st_content_input)
                st_labels = Variable(st_batch['labels']) # (12,5,9)
                if self.use_mart or self.story_dual:
                    st_input_ids = Variable(st_batch['input_ids'])
                    st_masks = Variable(st_batch['masks'])
                # Moda
                if self.use_segment:
                    se_real_cpu = im_batch['images_seg']
                    se_real_imgs = Variable(se_real_cpu)

                if self.cfg.CUDA and self.cuda_is_available: # Moda-fix: add cuda condition
                    st_real_imgs = st_real_imgs.cuda()
                    im_real_imgs = im_real_imgs.cuda()
                    st_motion_input = st_motion_input.cuda()
                    im_motion_input = im_motion_input.cuda()
                    st_content_input = st_content_input.cuda()
                    im_content_input = im_content_input.cuda()
                    im_labels = im_labels.cuda()
                    st_labels = st_labels.cuda()
                    if self.use_mart or self.img_dual:
                        im_input_ids = im_input_ids.cuda()
                        im_masks = im_masks.cuda()
                    if self.story_dual or self.use_mart:
                        st_input_ids = st_input_ids.cuda()
                        st_masks = st_masks.cuda()
                    # Moda
                    if self.use_segment:
                        se_real_imgs = se_real_imgs.cuda()

                im_motion_input = torch.cat((im_motion_input, im_labels), 1)
                st_motion_input = torch.cat((st_motion_input, st_labels), 2)
                #######################################################
                # (2) Generate fake stories and images
                ######################################################

                if len(self.gpus) > 1:
                    netG = nn.DataParallel(netG)

                if self.use_mart:
                    st_inputs = (st_motion_input, st_content_input, st_input_ids, st_masks, st_labels, self.use_segment)
                else:
                    st_inputs = (st_motion_input, st_content_input, self.use_segment)
                #lr_st_fake, st_fake, m_mu, m_logvar, c_mu, c_logvar = \
                #    nn.parallel.data_parallel(netG.sample_videos, st_inputs, self.gpus)

                # Moda: extra return (se_fake == _) from sample_videos
                lr_st_fake, st_fake, m_mu, m_logvar, c_mu, c_logvar, _ = netG.sample_videos(*st_inputs) # m_mu (60,365), c_mu (12,124)

                if self.use_mart:
                    im_inputs = (im_motion_input, im_content_input, im_input_ids, im_masks, im_labels)
                    # Moda
                    if self.use_segment: im_inputs += (self.use_segment,)
                else:
                    im_inputs = (im_motion_input, im_content_input)
                    # Moda
                    if self.use_segment: im_inputs += (self.use_segment,)
                #lr_im_fake, im_fake, im_mu, im_logvar, cim_mu, cim_logvar = \
                #    nn.parallel.data_parallel(netG.sample_images, im_inputs, self.gpus)

                # Moda: extra return (se_fake) from sample_images
                lr_im_fake, im_fake, im_mu, im_logvar, cim_mu, cim_logvar, se_fake = netG.sample_images(*im_inputs) # im_mu (60,489), cim_mu (60,124)

                characters_mu = (st_labels.mean(1)>0).type(torch.FloatTensor) # which character exists in the full story (5 descriptions)
                if self.cuda_is_available: characters_mu.cuda() # Moda-fix: add cuda condition
                st_mu = torch.cat((c_mu, st_motion_input[:,:, :self.cfg.TEXT.DIMENSION].mean(1).squeeze(), characters_mu), 1)

                im_mu = torch.cat((im_motion_input, cim_mu), 1)
                ############################
                # (3) Update D network
                ###########################
                if self.use_image_disc:
                    netD_im.zero_grad()
                    im_errD, imgD_loss_report = \
                        compute_discriminator_loss(netD_im, im_real_imgs, im_fake,
                                                   im_real_labels, im_fake_labels, im_labels,
                                                   im_mu, self.gpus, mode='image')
                    im_errD.backward()
                    im_optimizerD.step()
                else:
                    im_errD = torch.tensor(0)
                    imgD_loss_report = {}

                if self.use_story_disc:
                    Mode = 'story'
                    netD_st.zero_grad()
                    st_errD, stD_loss_report = \
                        compute_discriminator_loss(netD_st, st_real_imgs, st_fake,
                                                   st_real_labels, st_fake_labels, st_labels,
                                                   st_mu, self.gpus, mode='story')
                    st_errD.backward()
                    st_optimizerD.step()
                else:
                    st_errD = torch.tensor(0)
                    stD_loss_report = {}

                # Moda
                if self.use_segment:
                    netD_se.zero_grad()
                    se_errD, seD_loss_report = \
                        compute_discriminator_loss(netD_se, se_real_imgs, se_fake,
                                                   im_real_labels, im_fake_labels, im_labels,
                                                   im_mu, self.gpus, mode='image')
                    se_errD.backward()
                    se_optimizerD.step()
                else:
                    se_errD = torch.tensor(0)
                    seD_loss_report = {}

                ############################
                # (2) Update G network
                ###########################
                for g_iter in range(self.cfg.TRAIN.UPDATE_RATIO):
                    # Moda-XXX
                    if g_iter > 0: break
                    print(f">>> g_iter ::: {g_iter}/{self.cfg.TRAIN.UPDATE_RATIO}")
                    netG.zero_grad()
                    if self.use_mart:
                        st_inputs = (st_motion_input, st_content_input, st_input_ids, st_masks, st_labels, self.use_segment)
                    else:
                        st_inputs = (st_motion_input, st_content_input, self.use_segment)
                    #_, st_fake, m_mu, m_logvar, c_mu, c_logvar = \
                    #    nn.parallel.data_parallel(netG.sample_videos, st_inputs, self.gpus)

                    # Moda: extra return (se_fake == _) from sample_videos
                    video_latents, st_fake, m_mu, m_logvar, c_mu, c_logvar, _ = netG.sample_videos(*st_inputs)

                    if self.use_mart:
                        im_inputs = (im_motion_input, im_content_input, im_input_ids, im_masks, im_labels)
                        # Moda
                        if self.use_segment: im_inputs += (self.use_segment,)
                    else:
                        im_inputs = (im_motion_input, im_content_input)
                        # Moda
                        if self.use_segment: im_inputs += (self.use_segment,)
                    #_, im_fake, im_mu, im_logvar, cim_mu, cim_logvar = \
                    #nn.parallel.data_parallel(netG.sample_images, im_inputs, self.gpus)

                    # Moda: extra return (se_fake) from sample_images
                    image_latents, im_fake, im_mu, im_logvar, cim_mu, cim_logvar, se_fake = netG.sample_images(*im_inputs)

                    # Moda
                    # encoder_decoder_loss = 0 # Moda: not used
                    if video_latents is not None:
                        ((h_seg1, h_seg2, h_seg3, h_seg4), (g_seg1, g_seg2, g_seg3, g_seg4)) = video_latents

                        video_latent_loss = mse_loss(g_seg1, h_seg1) + mse_loss(g_seg2, h_seg2 ) + mse_loss(g_seg3, h_seg3) + mse_loss(g_seg4, h_seg4)
                        ((h_seg1, h_seg2, h_seg3, h_seg4), (g_seg1, g_seg2, g_seg3, g_seg4)) = image_latents
                        image_latent_loss = mse_loss(g_seg1, h_seg1) + mse_loss(g_seg2, h_seg2 ) + mse_loss(g_seg3, h_seg3) + mse_loss(g_seg4, h_seg4)
                        # encoder_decoder_loss = ( image_latent_loss + video_latent_loss ) / 2 # Moda: not used

                        reconstruct_img = netG.train_autoencoder(se_real_imgs)
                        reconstruct_fake = netG.train_autoencoder(se_fake)
                        reconstruct_loss = (mse_loss(reconstruct_img, se_real_imgs) + mse_loss(reconstruct_fake, se_fake)) / 2.0
                    # Moda: logging code from CP-CSV
                    #     self._logger.add_scalar('G/image_vae_loss', image_latent_loss.data, step)
                    #     self._logger.add_scalar('G/video_vae_loss', video_latent_loss.data, step)
                    #     self._logger.add_scalar('G/reconstruct_loss', reconstruct_loss.data, step)

                    characters_mu = (st_labels.mean(1)>0).type(torch.FloatTensor)
                    if self.cuda_is_available: characters_mu.cuda() # Moda-fix: add cuda condition
                    st_mu = torch.cat((c_mu, st_motion_input[:,:, :self.cfg.TEXT.DIMENSION].mean(1).squeeze(), characters_mu), 1)

                    im_mu = torch.cat((im_motion_input, cim_mu), 1)

                    # Moda: pass extra parameter to compute_generator_loss
                    if self.use_image_disc:
                        im_errG, imG_loss_report = compute_generator_loss(netD_im, im_fake, im_real_imgs,
                                                               im_real_labels, im_labels, im_mu, self.gpus,
                                                               mode='image')
                    else:
                        im_errG = torch.tensor(0)
                        imG_loss_report = {}

                    if self.use_story_disc:
                        st_errG, stG_loss_report = compute_generator_loss(netD_st, st_fake, st_real_imgs,
                                                            st_real_labels, st_labels, st_mu, self.gpus,
                                                            mode='story')
                    else:
                        st_errG = torch.tensor(0)
                        stG_loss_report = {}


                    if self.use_segment:
                        se_errG, seG_loss_report = compute_generator_loss(netD_se, se_fake, se_real_imgs,
                                                            im_real_labels, im_labels, im_mu, self.gpus,
                                                            mode='image')
                    else:
                        se_errG = torch.tensor(0)
                        seG_loss_report = {}

                    # Moda
                    ## TODO: Check if Seg is included here
                    if self.cfg.STORY_DUAL:
                        st_errDual, st_videocap_loss_report = compute_dual_captioning_loss(netDual, st_fake, (st_input_ids, st_masks),
                                                                                           storyloader.dataset.vocab, self.gpus, img_ft_extractor, transform)
                    
                    ######
                    # Sample Image Loss and Sample Video Loss
                    im_kl_loss = KL_loss(cim_mu, cim_logvar)
                    st_kl_loss = KL_loss(c_mu, c_logvar)

                    # Moda
                    ## TODO: make sure that kl_loss comment is correct
                    # errG =  im_errG + self.ratio * ( image_weight*st_errG + se_errG*segment_weight) # for record
                    # kl_loss = im_kl_loss + self.ratio * st_kl_loss # for record

                    # Moda : Replace the cond tree by one line
                    ## TODO: Check image_weight and segment_weight?
                    errG_total = im_errG + im_kl_loss * self.cfg.TRAIN.COEFF.KL + \
                                self.ratio * (se_errG*segment_weight + st_errG*image_weight + st_kl_loss * self.cfg.TRAIN.COEFF.KL)
                    # if self.use_image_disc and self.use_story_disc:
                    #     errG_total = im_errG + im_kl_loss * self.cfg.TRAIN.COEFF.KL + self.ratio * (
                    #             st_errG + st_kl_loss * self.cfg.TRAIN.COEFF.KL)
                    # elif self.use_image_disc:
                    #     errG_total = im_errG + im_kl_loss * self.cfg.TRAIN.COEFF.KL + self.ratio * (st_kl_loss * self.cfg.TRAIN.COEFF.KL)
                    # else:
                    #     errG_total = im_kl_loss * self.cfg.TRAIN.COEFF.KL + self.ratio * (st_errG + st_kl_loss * self.cfg.TRAIN.COEFF.KL)

                    if self.cfg.TRAIN.PERCEPTUAL_LOSS:
                        if self.cfg.CUDA and self.cuda_is_available: # Moda-fix: add cuda condition
                            per_loss = self.perceptual_loss_net(im_fake, im_real_cpu.cuda())
                        else:
                            per_loss = self.perceptual_loss_net(im_fake, im_real_cpu)
                        errG_total += per_loss

                    if self.cfg.STORY_DUAL:
                        errG_total += st_errDual

                    # Moda
                    if video_latents is not None:
                        errG_total += ( video_latent_loss +  reconstruct_loss )* cfg.RECONSTRUCT_LOSS

                    errG_total.backward()
                    optimizerG.step()

                # delete variables to free space?
                del st_real_imgs, im_real_imgs, st_motion_input, im_motion_input, st_content_input, im_content_input, im_labels, st_labels
                if self.use_mart or self.img_dual:
                    del im_input_ids, im_masks
                if self.story_dual or self.use_mart:
                    del st_input_ids, st_masks

                # if i%20 == 0 and i>0:
                #     save_test_samples(netG, testloader, self.test_dir, epoch, mart=self.use_mart)

                # Moda
                loss_collector.append([imgD_loss_report, imG_loss_report, stD_loss_report, stG_loss_report, seD_loss_report, seG_loss_report])
                count = count + 1

            end_t = time.time()
            print('''[%d/%d][%d/%d] %s Total Time: %.2fsec'''
                  % (epoch, self.max_epoch, i, len(storyloader), cfg.DATASET_NAME, (end_t - start_t)))

            # Moda
            for loss_report in [imgD_loss_report, imG_loss_report, stD_loss_report, stG_loss_report, seD_loss_report, seG_loss_report]:
                for key, val in loss_report.items():
                    print(key, val)
            if self.cfg.TRAIN.PERCEPTUAL_LOSS:
                print("Perceptual Loss: ", per_loss.data.item())
            if self.cfg.STORY_DUAL:
                for key, val in st_videocap_loss_report.items():
                    print(key, val)

            print('--------------------------------------------------------------------------------')

            if epoch % self.snapshot_interval == 0:
                save_test_samples(netG, testloader, self.test_dir, epoch, mart=self.use_mart, seg=self.use_segment)
            if epoch % 5 == 0:
                save_model(netG, netD_im, netD_st, netD_se, epoch, self.model_dir)

        with open(os.path.join(self.model_dir, 'losses.pkl'), 'wb') as f:
            pickle.dump(loss_collector, f)

        # Moda
        save_model(netG, netD_im, netD_st, netD_se, self.max_epoch, self.model_dir)

    def sample(self, testloader, generator_weight_path, out_dir, stage=1):

        if stage == 1:
            netG, _, _, _ = self.load_network_stageI(self.use_mart, self.use_segment) # Moda: extra return
        else:
            raise ValueError
        netG.load_state_dict(torch.load(generator_weight_path))
        save_test_samples(netG, testloader, out_dir, 60, mart=self.use_mart, seg=self.use_segment)
