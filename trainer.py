# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
from decimal import Decimal
from torch.autograd import Variable
import utility
import torch
import torch.nn.utils as utils
from tqdm import tqdm

from loss.GeneratorLoss import GeneratorLoss
from model.Discriminator import Discriminator
import torch.optim as optim


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test

        # 1.声明网络
        self.model = my_model
        self.modelD = Discriminator()

        # 2.声明优化器目标
        self.generator_criterion = GeneratorLoss()

        self.loss = my_loss

        # 3.生成器、判别器优化器
        self.optimizer = utility.make_optimizer(args, self.model)
        self.optimizerD = utility.make_optimizer(args, self.modelD)

        # 4.使用GPU
        self.modelD.cuda()
        self.model.cuda()

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):

        # self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        # self.loss.start_log()

        # 5.生成器、判别器模型训练
        self.model.train()
        self.modelD.train()

        timer_data, timer_model = utility.timer(), utility.timer()

        # TEMP
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            # 6.优化判别器
            real_img = Variable(hr).cuda()
            z = Variable(lr).cuda()
            fake_img = self.model(z, 0)

            self.optimizerD.zero_grad()
            real_out = self.modelD(real_img).mean()
            fake_out = self.modelD(fake_img).mean()
            d_loss = 1 - real_out + fake_out

            d_loss.backward(retain_graph=True)

            # 7.优化生成器
            self.optimizer.zero_grad()
            g_loss = self.generator_criterion(fake_out, fake_img, real_img)
            g_loss.backward()

            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            # g_loss = 1
            # d_loss = 1
            self.optimizerD.step()
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t[g_loss]{}\t[d_loss]{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    g_loss,
                    d_loss,
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        # self.loss.end_log(len(self.loader_train))
        # self.error_last = self.loss.log[-1, -1]

        self.optimizer.schedule()
        self.optimizerD.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()
        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            i = 0
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                if self.args.derain:
                    for norain, rain, filename in tqdm(d, ncols=80):
                        norain, rain = self.prepare(norain, rain)
                        sr = self.model(rain, idx_scale)
                        sr = utility.quantize(sr, self.args.rgb_range)

                        save_list = [sr]
                        self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                            sr, norain, scale, self.args.rgb_range
                        )
                        if self.args.save_results:
                            self.ckp.save_results(d, filename[0], save_list, 1)
                    self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                    best = self.ckp.log.max(0)
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, idx_data, idx_scale],
                            best[0][idx_data, idx_scale],
                            best[1][idx_data, idx_scale] + 1
                        )
                    )
                    isderain = 0
                elif self.args.denoise:
                    for hr, _, filename in tqdm(d, ncols=80):
                        hr = self.prepare(hr)[0]
                        noisy_level = self.args.sigma
                        noise = torch.randn(hr.size()).mul_(noisy_level).cuda()
                        nois_hr = (noise + hr).clamp(0, 255)
                        sr = self.model(nois_hr, idx_scale)
                        sr = utility.quantize(sr, self.args.rgb_range)

                        save_list = [sr, nois_hr, hr]
                        self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range
                        )
                        if self.args.save_results:
                            self.ckp.save_results(d, filename[0], save_list, 50)

                    self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                    best = self.ckp.log.max(0)
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, idx_data, idx_scale],
                            best[0][idx_data, idx_scale],
                            best[1][idx_data, idx_scale] + 1
                        )
                    )
                else:
                    for lr, hr, filename in tqdm(d, ncols=80):
                        lr, hr = self.prepare(lr, hr)
                        sr = self.model(lr, idx_scale)
                        sr = utility.quantize(sr, self.args.rgb_range)

                        save_list = [sr]
                        self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range
                        )
                        # self.ckp.log[-1, idx_data, idx_scale] += utility.calc_ssim(
                        #     sr, hr
                        # )
                        # import pdb
                        # pdb.set_trace()
                        if self.args.save_gt:
                            save_list.extend([lr, hr])

                        if self.args.save_results:
                            self.ckp.save_results(d, filename[0], save_list, scale)
                        i = i + 1
                    self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                    best = self.ckp.log.max(0)
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, idx_data, idx_scale],
                            best[0][idx_data, idx_scale],
                            best[1][idx_data, idx_scale] + 1
                        )
                    )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

            if not self.args.test_only:
                self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs
