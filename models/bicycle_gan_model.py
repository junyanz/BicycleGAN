import numpy as np
import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from .base_model import BaseModel
from . import networks
import random


class BiCycleGANModel(BaseModel):
    def name(self):
        return 'BiCycleGANModel'

    def initialize(self, opt):
        if opt.isTrain:
            assert opt.batchSize % 2 == 0  # load two images at one time.

        use_D = opt.isTrain and opt.lambda_GAN > 0.0
        use_D2 = opt.isTrain and opt.lambda_GAN2 > 0.0 and not opt.use_same_D
        use_E = opt.isTrain or not opt.no_encode
        BaseModel.initialize(self, opt)
        self.init_data(opt, use_D=use_D, use_D2=use_D2, use_E=use_E, use_vae=True)
        self.skip = False

    def is_skip(self):
        return self.skip

    def forward(self):
        # get real images
        self.skip = self.opt.isTrain and self.input_A.size(0) < self.opt.batchSize
        if self.skip:
            print('skip this point data_size = %d' % self.input_A.size(0))
            return
        half_size = self.opt.batchSize // 2
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        # A1, B1 for encoded; A2, B2 for random
        self.real_A_encoded = self.real_A[0:half_size]
        self.real_A_random = self.real_A[half_size:]
        self.real_B_encoded = self.real_B[0:half_size]
        self.real_B_random = self.real_B[half_size:]
        # get encoded z

        self.mu, self.logvar = self.netE.forward(self.real_B_encoded)
        std = self.logvar.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1), 'gauss')
        self.z_encoded = eps.mul(std).add_(self.mu)
        # get random z
        self.z_random = self.get_z_random(self.real_A_random.size(0), self.opt.nz, 'gauss')
        # generate fake_B_encoded
        self.fake_B_encoded = self.netG.forward(self.real_A_encoded, self.z_encoded)
        # generate fake_B_random
        self.fake_B_random = self.netG.forward(self.real_A_encoded, self.z_random)
        if self.opt.conditional_D:   # tedious conditoinal data
            self.fake_data_encoded = torch.cat([self.real_A_encoded, self.fake_B_encoded], 1)
            self.real_data_encoded = torch.cat([self.real_A_encoded, self.real_B_encoded], 1)
            self.fake_data_random = torch.cat([self.real_A_encoded, self.fake_B_random], 1)
            self.real_data_random = torch.cat([self.real_A_random, self.real_B_random], 1)
        else:
            self.fake_data_encoded = self.fake_B_encoded
            self.fake_data_random = self.fake_B_random
            self.real_data_encoded = self.real_B_encoded
            self.real_data_random = self.real_B_random

        # compute z_predict
        if self.opt.lambda_z > 0.0:
            self.mu2, logvar2 = self.netE.forward(self.fake_B_random)
            self.std2 = logvar2.mul(0.5).exp_()
            eps2 = self.get_z_random(self.std2.size(0), self.std2.size(1), 'gauss')
            self.z_predict = eps2.mul(self.std2).add_(self.mu2)

    def encode(self, input_data):
        mu, logvar = self.netE.forward(Variable(input_data, volatile=True))
        std = logvar.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1), 'gauss')
        return eps.mul(std).add_(mu)

    def backward_D(self, netD, real, fake):
        # Fake, stop backprop to the generator by detaching fake_B
        pred_fake = netD.forward(fake.detach())
        # real
        pred_real = netD.forward(real)
        loss_D_fake, losses_D_fake = self.criterionGAN(pred_fake, False)
        loss_D_real, losses_D_real = self.criterionGAN(pred_real, True)
        # Combined loss
        loss_D = loss_D_fake + loss_D_real
        loss_D.backward()
        return loss_D, [loss_D_fake, loss_D_real]

    def backward_G_GAN(self, fake, netD=None, ll=0.0):
        if ll > 0.0:
            pred_fake = netD.forward(fake)
            loss_G_GAN, losses_G_GAN = self.criterionGAN(pred_fake, True)
        else:
            loss_G_GAN = 0
        return loss_G_GAN * ll

    def backward_EG(self):
        # 1, G(A) should fool D
        self.loss_G_GAN = self.backward_G_GAN(self.fake_data_encoded, self.netD, self.opt.lambda_GAN)
        if self.opt.use_same_D:
            self.loss_G_GAN2 = self.backward_G_GAN(self.fake_data_random, self.netD, self.opt.lambda_GAN2)
        else:
            self.loss_G_GAN2 = self.backward_G_GAN(self.fake_data_random, self.netD2, self.opt.lambda_GAN2)
        # 2. KL loss
        if self.opt.lambda_kl > 0.0:
            kl_element = self.mu.pow(2).add_(self.logvar.exp()).mul_(-1).add_(1).add_(self.logvar)
            self.loss_kl = torch.sum(kl_element).mul_(-0.5) * self.opt.lambda_kl
        else:
            self.loss_kl = 0
        # 3, reconstruction |fake_B-real_B|
        if self.opt.lambda_L1 > 0.0:
            self.loss_G_L1 = self.criterionL1(self.fake_B_encoded, self.real_B_encoded) * self.opt.lambda_L1
        else:
            self.loss_G_L1 = 0.0

        self.loss_G = self.loss_G_GAN + self.loss_G_GAN2 + self.loss_G_L1 + self.loss_kl
        self.loss_G.backward(retain_graph=True)

    def update_D(self, data):
        self.set_requires_grad(self.netD, True)
        self.set_input(data)
        self.forward()
        if self.is_skip():
            return
        # update D1
        if self.opt.lambda_GAN > 0.0:
            self.optimizer_D.zero_grad()
            self.loss_D, self.losses_D = self.backward_D(self.netD, self.real_data_encoded, self.fake_data_encoded)
            if self.opt.use_same_D:
                self.loss_D2, self.losses_D2 = self.backward_D(self.netD, self.real_data_random, self.fake_data_random)
            self.optimizer_D.step()

        if self.opt.lambda_GAN2 > 0.0 and not self.opt.use_same_D:
            self.optimizer_D2.zero_grad()
            self.loss_D2, self.losses_D2 = self.backward_D(self.netD2, self.real_data_random, self.fake_data_random)
            self.optimizer_D2.step()

    def backward_G_alone(self):
        # 3, reconstruction |z_predit-z_random|
        if self.opt.lambda_z > 0.0:
            self.loss_z_L1 = torch.mean(torch.abs(self.mu2 - self.z_random)) * self.opt.lambda_z
            self.loss_z_L1.backward()
        else:
            self.loss_z_L1 = 0.0

    def update_G(self):
        # update G and E
        self.set_requires_grad(self.netD, False)
        self.optimizer_E.zero_grad()
        self.optimizer_G.zero_grad()
        self.backward_EG()
        self.optimizer_G.step()
        self.optimizer_E.step()
        # update G only
        if self.opt.lambda_z > 0.0:
            self.optimizer_G.zero_grad()
            self.optimizer_E.zero_grad()
            self.backward_G_alone()
            self.optimizer_G.step()

    def get_current_errors(self):
        z1 = self.z_encoded.data.cpu().numpy()
        if self.opt.lambda_z > 0.0:
            loss_G = self.loss_G + self.loss_z_L1
        else:
            loss_G = self.loss_G
        ret_dict = OrderedDict([('z_encoded_mag', np.mean(np.abs(z1))),
                                ('G_total', loss_G.data[0])])

        if self.opt.lambda_L1 > 0.0:
            G_L1 = self.loss_G_L1.data[0] if self.loss_G_L1 is not None else 0.0
            ret_dict['G_L1_encoded'] = self.loss_G_L1.data[0]

        if self.opt.lambda_z > 0.0:
            z_L1 = self.loss_z_L1.data[0] if self.loss_z_L1 is not None else 0.0
            ret_dict['z_L1'] = z_L1

        if self.opt.lambda_kl > 0.0:
            ret_dict['KL'] = self.loss_kl.data[0]

        if self.opt.lambda_GAN > 0.0:
            ret_dict['G_GAN'] = self.loss_G_GAN.data[0]
            ret_dict['D_GAN'] = self.loss_D.data[0]

        if self.opt.lambda_GAN2 > 0.0:
            ret_dict['G_GAN2'] = self.loss_G_GAN2.data[0]
            ret_dict['D_GAN2'] = self.loss_D2.data[0]
        return ret_dict

    def get_current_visuals(self):
        real_A_encoded = util.tensor2im(self.real_A_encoded.data)
        real_A_random = util.tensor2im(self.real_A_random.data)
        real_B_encoded = util.tensor2im(self.real_B_encoded.data)
        real_B_random = util.tensor2im(self.real_B_random.data)

        if self.opt.isTrain:
            fake_random = util.tensor2im(self.fake_B_random.data)
            fake_encoded = util.tensor2im(self.fake_B_encoded.data)
            ret_dict = OrderedDict([('real_A_encoded', real_A_encoded),  ('real_B_encoded', real_B_encoded), ('fake_B_encoded', fake_encoded),
                                    ('fake_B_random', fake_random), ('real_A_random', real_A_random), ('real_B_random', real_B_random)])
            return ret_dict

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        if self.opt.lambda_GAN > 0.0:
            self.save_network(self.netD, 'D', label, self.gpu_ids)
        if self.opt.lambda_GAN2 > 0.0 and not self.opt.use_same_D:
            self.save_network(self.netD, 'D2', label, self.gpu_ids)
        self.save_network(self.netE, 'E', label, self.gpu_ids)
