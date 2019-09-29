# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

from run_manager import *


class ArchSearchConfig:

    def __init__(self, arch_init_type, arch_init_ratio, arch_opt_type, arch_lr,
                 arch_opt_param, arch_weight_decay, target_hardware, ref_value):
        """ architecture parameters initialization & optimizer """
        self.arch_init_type = arch_init_type
        self.arch_init_ratio = arch_init_ratio

        self.opt_type = arch_opt_type
        self.lr = arch_lr
        self.opt_param = {} if arch_opt_param is None else arch_opt_param
        self.weight_decay = arch_weight_decay
        self.target_hardware = target_hardware
        self.ref_value = ref_value

    @property
    def config(self):
        config = {
            'type': type(self),
        }
        for key in self.__dict__:
            if not key.startswith('_'):
                config[key] = self.__dict__[key]
        return config

    def get_update_schedule(self, nBatch):
        raise NotImplementedError

    def build_optimizer(self, params):
        """

        :param params: architecture parameters
        :return: arch_optimizer
        """
        if self.opt_type == 'adam':
            return torch.optim.Adam(
                params, self.lr, weight_decay=self.weight_decay, **self.opt_param
            )
        else:
            raise NotImplementedError


class GradientArchSearchConfig(ArchSearchConfig):

    def __init__(self, arch_init_type='normal', arch_init_ratio=1e-3, arch_opt_type='adam', arch_lr=1e-3,
                 arch_opt_param=None, arch_weight_decay=0, target_hardware=None, ref_value=None,
                 grad_update_arch_param_every=1, grad_update_steps=1, grad_binary_mode='full', grad_data_batch=None,
                 grad_reg_loss_type=None, grad_reg_loss_params=None, **kwargs):
        super(GradientArchSearchConfig, self).__init__(
            arch_init_type, arch_init_ratio, arch_opt_type, arch_lr, arch_opt_param, arch_weight_decay,
            target_hardware, ref_value,
        )

        self.update_arch_param_every = grad_update_arch_param_every
        self.update_steps = grad_update_steps
        self.binary_mode = grad_binary_mode
        self.data_batch = grad_data_batch

        self.reg_loss_type = grad_reg_loss_type
        self.reg_loss_params = {} if grad_reg_loss_params is None else grad_reg_loss_params

        print(kwargs.keys())

    def get_update_schedule(self, nBatch):
        schedule = {}
        for i in range(nBatch):
            if (i + 1) % self.update_arch_param_every == 0:
                schedule[i] = self.update_steps
        return schedule

    def add_regularization_loss(self, ce_loss, expected_value):
        if expected_value is None:
            return ce_loss

        if self.reg_loss_type == 'mul#log':
            alpha = self.reg_loss_params.get('alpha', 1)
            beta = self.reg_loss_params.get('beta', 0.6)
            # noinspection PyUnresolvedReferences
            reg_loss = (torch.log(expected_value) / math.log(self.ref_value)) ** beta
            return alpha * ce_loss * reg_loss
        elif self.reg_loss_type == 'add#linear':
            reg_lambda = self.reg_loss_params.get('lambda', 2e-1)
            reg_loss = reg_lambda * (expected_value - self.ref_value) / self.ref_value
            return ce_loss + reg_loss
        elif self.reg_loss_type is None:
            return ce_loss
        else:
            raise ValueError('Do not support: %s' % self.reg_loss_type)


class RLArchSearchConfig(ArchSearchConfig):

    def __init__(self, arch_init_type='normal', arch_init_ratio=1e-3, arch_opt_type='adam', arch_lr=1e-3,
                 arch_opt_param=None, arch_weight_decay=0, target_hardware=None, ref_value=None,
                 rl_batch_size=10, rl_update_per_epoch=False, rl_update_steps_per_epoch=300,
                 rl_baseline_decay_weight=0.99, rl_tradeoff_ratio=0.1, **kwargs):
        super(RLArchSearchConfig, self).__init__(
            arch_init_type, arch_init_ratio, arch_opt_type, arch_lr, arch_opt_param, arch_weight_decay,
            target_hardware, ref_value,
        )

        self.batch_size = rl_batch_size
        self.update_per_epoch = rl_update_per_epoch
        self.update_steps_per_epoch = rl_update_steps_per_epoch
        self.baseline_decay_weight = rl_baseline_decay_weight
        self.tradeoff_ratio = rl_tradeoff_ratio

        self._baseline = None
        print(kwargs.keys())

    def get_update_schedule(self, nBatch):
        schedule = {}
        if self.update_per_epoch:
            schedule[nBatch - 1] = self.update_steps_per_epoch
        else:
            rl_seg_list = get_split_list(nBatch, self.update_steps_per_epoch)
            for j in range(1, len(rl_seg_list)):
                rl_seg_list[j] += rl_seg_list[j - 1]
            for j in rl_seg_list:
                schedule[j - 1] = 1
        return schedule

    def calculate_reward(self, net_info):
        acc = net_info['acc'] / 100
        if self.target_hardware is None:
            return acc
        else:
            return acc * ((self.ref_value / net_info[self.target_hardware]) ** self.tradeoff_ratio)

    @property
    def baseline(self):
        return self._baseline

    @baseline.setter
    def baseline(self, value):
        self._baseline = value


class ArchSearchRunManager:

    def __init__(self, path, super_net, run_config: RunConfig, arch_search_config: ArchSearchConfig):
        # init weight parameters & build weight_optimizer
        self.run_manager = RunManager(path, super_net, run_config, True)

        self.arch_search_config = arch_search_config

        # init architecture parameters
        self.net.init_arch_params(
            self.arch_search_config.arch_init_type, self.arch_search_config.arch_init_ratio,
        )

        # build architecture optimizer
        self.arch_optimizer = self.arch_search_config.build_optimizer(self.net.architecture_parameters())

        self.warmup = True
        self.warmup_epoch = 0

    @property
    def net(self):
        return self.run_manager.net.module

    def write_log(self, log_str, prefix, should_print=True, end='\n'):
        with open(os.path.join(self.run_manager.logs_path, '%s.log' % prefix), 'a') as fout:
            fout.write(log_str + end)
            fout.flush()
        if should_print:
            print(log_str)

    def load_model(self, model_fname=None):
        latest_fname = os.path.join(self.run_manager.save_path, 'latest.txt')
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, 'r') as fin:
                model_fname = fin.readline()
                if model_fname[-1] == '\n':
                    model_fname = model_fname[:-1]

        if model_fname is None or not os.path.exists(model_fname):
            model_fname = '%s/checkpoint.pth.tar' % self.run_manager.save_path
            with open(latest_fname, 'w') as fout:
                fout.write(model_fname + '\n')
        if self.run_manager.out_log:
            print("=> loading checkpoint '{}'".format(model_fname))

        if torch.cuda.is_available():
            checkpoint = torch.load(model_fname)
        else:
            checkpoint = torch.load(model_fname, map_location='cpu')

        model_dict = self.net.state_dict()
        model_dict.update(checkpoint['state_dict'])
        self.net.load_state_dict(model_dict)
        if self.run_manager.out_log:
            print("=> loaded checkpoint '{}'".format(model_fname))

        # set new manual seed
        new_manual_seed = int(time.time())
        torch.manual_seed(new_manual_seed)
        torch.cuda.manual_seed_all(new_manual_seed)
        np.random.seed(new_manual_seed)

        if 'epoch' in checkpoint:
            self.run_manager.start_epoch = checkpoint['epoch'] + 1
        if 'weight_optimizer' in checkpoint:
            self.run_manager.optimizer.load_state_dict(checkpoint['weight_optimizer'])
        if 'arch_optimizer' in checkpoint:
            self.arch_optimizer.load_state_dict(checkpoint['arch_optimizer'])
        if 'warmup' in checkpoint:
            self.warmup = checkpoint['warmup']
        if self.warmup and 'warmup_epoch' in checkpoint:
            self.warmup_epoch = checkpoint['warmup_epoch']

    """ training related methods """

    def validate(self):
        # get performances of current chosen network on validation set
        self.run_manager.run_config.valid_loader.batch_sampler.batch_size = self.run_manager.run_config.test_batch_size
        self.run_manager.run_config.valid_loader.batch_sampler.drop_last = False

        # set chosen op active
        self.net.set_chosen_op_active()
        # remove unused modules
        self.net.unused_modules_off()
        # test on validation set under train mode
        valid_res = self.run_manager.validate(is_test=False, use_train_mode=True, return_top5=True)
        # flops of chosen network
        flops = self.run_manager.net_flops()
        # measure latencies of chosen op
        if self.arch_search_config.target_hardware in [None, 'flops']:
            latency = 0
        else:
            latency, _ = self.run_manager.net_latency(
                l_type=self.arch_search_config.target_hardware, fast=False
            )
        # unused modules back
        self.net.unused_modules_back()
        return valid_res, flops, latency

    def warm_up(self, warmup_epochs=25):
        lr_max = 0.05
        data_loader = self.run_manager.run_config.train_loader
        nBatch = len(data_loader)
        T_total = warmup_epochs * nBatch

        for epoch in range(self.warmup_epoch, warmup_epochs):
            print('\n', '-' * 30, 'Warmup epoch: %d' % (epoch + 1), '-' * 30, '\n')
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            # switch to train mode
            self.run_manager.net.train()

            end = time.time()
            for i, (images, labels) in enumerate(data_loader):
                data_time.update(time.time() - end)
                # lr
                T_cur = epoch * nBatch + i
                warmup_lr = 0.5 * lr_max * (1 + math.cos(math.pi * T_cur / T_total))
                for param_group in self.run_manager.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                images, labels = images.to(self.run_manager.device), labels.to(self.run_manager.device)
                # compute output
                self.net.reset_binary_gates()  # random sample binary gates
                self.net.unused_modules_off()  # remove unused module for speedup
                output = self.run_manager.net(images)  # forward (DataParallel)
                # loss
                if self.run_manager.run_config.label_smoothing > 0:
                    loss = cross_entropy_with_label_smoothing(
                        output, labels, self.run_manager.run_config.label_smoothing
                    )
                else:
                    loss = self.run_manager.criterion(output, labels)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                losses.update(loss, images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                # compute gradient and do SGD step
                self.run_manager.net.zero_grad()  # zero grads of weight_param, arch_param & binary_param
                loss.backward()
                self.run_manager.optimizer.step()  # update weight parameters
                # unused modules back
                self.net.unused_modules_back()
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.run_manager.run_config.print_frequency == 0 or i + 1 == nBatch:
                    batch_log = 'Warmup Train [{0}][{1}/{2}]\t' \
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                                'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                                'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})\t' \
                                'Top-5 acc {top5.val:.3f} ({top5.avg:.3f})\tlr {lr:.5f}'. \
                        format(epoch + 1, i, nBatch - 1, batch_time=batch_time, data_time=data_time,
                               losses=losses, top1=top1, top5=top5, lr=warmup_lr)
                    self.run_manager.write_log(batch_log, 'train')
            valid_res, flops, latency = self.validate()
            val_log = 'Warmup Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f}\ttop-5 acc {4:.3f}\t' \
                      'Train top-1 {top1.avg:.3f}\ttop-5 {top5.avg:.3f}\tflops: {5:.1f}M'. \
                format(epoch + 1, warmup_epochs, *valid_res, flops / 1e6, top1=top1, top5=top5)
            if self.arch_search_config.target_hardware not in [None, 'flops']:
                val_log += '\t' + self.arch_search_config.target_hardware + ': %.3fms' % latency
            self.run_manager.write_log(val_log, 'valid')
            self.warmup = epoch + 1 < warmup_epochs

            state_dict = self.net.state_dict()
            # rm architecture parameters & binary gates
            for key in list(state_dict.keys()):
                if 'AP_path_alpha' in key or 'AP_path_wb' in key:
                    state_dict.pop(key)
            checkpoint = {
                'state_dict': state_dict,
                'warmup': self.warmup,
            }
            if self.warmup:
                checkpoint['warmup_epoch'] = epoch,
            self.run_manager.save_model(checkpoint, model_name='warmup.pth.tar')

    def train(self, fix_net_weights=False):
        data_loader = self.run_manager.run_config.train_loader
        nBatch = len(data_loader)
        if fix_net_weights:
            data_loader = [(0, 0)] * nBatch

        arch_param_num = len(list(self.net.architecture_parameters()))
        binary_gates_num = len(list(self.net.binary_gates()))
        weight_param_num = len(list(self.net.weight_parameters()))
        print(
            '#arch_params: %d\t#binary_gates: %d\t#weight_params: %d' %
            (arch_param_num, binary_gates_num, weight_param_num)
        )

        update_schedule = self.arch_search_config.get_update_schedule(nBatch)

        for epoch in range(self.run_manager.start_epoch, self.run_manager.run_config.n_epochs):
            print('\n', '-' * 30, 'Train epoch: %d' % (epoch + 1), '-' * 30, '\n')
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            entropy = AverageMeter()
            # switch to train mode
            self.run_manager.net.train()

            end = time.time()
            for i, (images, labels) in enumerate(data_loader):
                data_time.update(time.time() - end)
                # lr
                lr = self.run_manager.run_config.adjust_learning_rate(
                    self.run_manager.optimizer, epoch, batch=i, nBatch=nBatch
                )
                # network entropy
                net_entropy = self.net.entropy()
                entropy.update(net_entropy.data.item() / arch_param_num, 1)
                # train weight parameters if not fix_net_weights
                if not fix_net_weights:
                    images, labels = images.to(self.run_manager.device), labels.to(self.run_manager.device)
                    # compute output
                    self.net.reset_binary_gates()  # random sample binary gates
                    self.net.unused_modules_off()  # remove unused module for speedup
                    output = self.run_manager.net(images)  # forward (DataParallel)
                    # loss
                    if self.run_manager.run_config.label_smoothing > 0:
                        loss = cross_entropy_with_label_smoothing(
                            output, labels, self.run_manager.run_config.label_smoothing
                        )
                    else:
                        loss = self.run_manager.criterion(output, labels)
                    # measure accuracy and record loss
                    acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                    losses.update(loss, images.size(0))
                    top1.update(acc1[0], images.size(0))
                    top5.update(acc5[0], images.size(0))
                    # compute gradient and do SGD step
                    self.run_manager.net.zero_grad()  # zero grads of weight_param, arch_param & binary_param
                    loss.backward()
                    self.run_manager.optimizer.step()  # update weight parameters
                    # unused modules back
                    self.net.unused_modules_back()
                # skip architecture parameter updates in the first epoch
                if epoch > 0:
                    # update architecture parameters according to update_schedule
                    for j in range(update_schedule.get(i, 0)):
                        start_time = time.time()
                        if isinstance(self.arch_search_config, RLArchSearchConfig):
                            reward_list, net_info_list = self.rl_update_step(fast=True)
                            used_time = time.time() - start_time
                            log_str = 'REINFORCE [%d-%d]\tTime %.4f\tMean Reward %.4f\t%s' % (
                                epoch + 1, i, used_time, sum(reward_list) / len(reward_list), net_info_list
                            )
                            self.write_log(log_str, prefix='rl', should_print=False)
                        elif isinstance(self.arch_search_config, GradientArchSearchConfig):
                            arch_loss, exp_value = self.gradient_step()
                            used_time = time.time() - start_time
                            log_str = 'Architecture [%d-%d]\t Time %.4f\t Loss %.4f\t %s %s' % \
                                      (epoch + 1, i, used_time, arch_loss,
                                       self.arch_search_config.target_hardware, exp_value)
                            self.write_log(log_str, prefix='gradient', should_print=False)
                        else:
                            raise ValueError('do not support: %s' % type(self.arch_search_config))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                # training log
                if i % self.run_manager.run_config.print_frequency == 0 or i + 1 == nBatch:
                    batch_log = 'Train [{0}][{1}/{2}]\t' \
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                                'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                                'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                                'Entropy {entropy.val:.5f} ({entropy.avg:.5f})\t' \
                                'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})\t' \
                                'Top-5 acc {top5.val:.3f} ({top5.avg:.3f})\tlr {lr:.5f}'. \
                        format(epoch + 1, i, nBatch - 1, batch_time=batch_time, data_time=data_time,
                               losses=losses, entropy=entropy, top1=top1, top5=top5, lr=lr)
                    self.run_manager.write_log(batch_log, 'train')

            # print current network architecture
            self.write_log('-' * 30 + 'Current Architecture [%d]' % (epoch + 1) + '-' * 30, prefix='arch')
            for idx, block in enumerate(self.net.blocks):
                self.write_log('%d. %s' % (idx, block.module_str), prefix='arch')
            self.write_log('-' * 60, prefix='arch')

            # validate
            if (epoch + 1) % self.run_manager.run_config.validation_frequency == 0:
                (val_loss, val_top1, val_top5), flops, latency = self.validate()
                self.run_manager.best_acc = max(self.run_manager.best_acc, val_top1)
                val_log = 'Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f} ({4:.3f})\ttop-5 acc {5:.3f}\t' \
                          'Train top-1 {top1.avg:.3f}\ttop-5 {top5.avg:.3f}\t' \
                          'Entropy {entropy.val:.5f}\t' \
                          'Latency-{6}: {7:.3f}ms\tFlops: {8:.2f}M'. \
                    format(epoch + 1, self.run_manager.run_config.n_epochs, val_loss, val_top1,
                           self.run_manager.best_acc, val_top5, self.arch_search_config.target_hardware,
                           latency, flops / 1e6, entropy=entropy, top1=top1, top5=top5)
                self.run_manager.write_log(val_log, 'valid')
            # save model
            self.run_manager.save_model({
                'warmup': False,
                'epoch': epoch,
                'weight_optimizer': self.run_manager.optimizer.state_dict(),
                'arch_optimizer': self.arch_optimizer.state_dict(),
                'state_dict': self.net.state_dict()
            })

        # convert to normal network according to architecture parameters
        normal_net = self.net.cpu().convert_to_normal_net()
        print('Total training params: %.2fM' % (count_parameters(normal_net) / 1e6))
        os.makedirs(os.path.join(self.run_manager.path, 'learned_net'), exist_ok=True)
        json.dump(normal_net.config, open(os.path.join(self.run_manager.path, 'learned_net/net.config'), 'w'), indent=4)
        json.dump(
            self.run_manager.run_config.config,
            open(os.path.join(self.run_manager.path, 'learned_net/run.config'), 'w'), indent=4,
        )
        torch.save(
            {'state_dict': normal_net.state_dict(), 'dataset': self.run_manager.run_config.dataset},
            os.path.join(self.run_manager.path, 'learned_net/init')
        )

    def rl_update_step(self, fast=True):
        assert isinstance(self.arch_search_config, RLArchSearchConfig)
        # prepare data
        self.run_manager.run_config.valid_loader.batch_sampler.batch_size = self.run_manager.run_config.test_batch_size
        self.run_manager.run_config.valid_loader.batch_sampler.drop_last = True
        # switch to train mode
        self.run_manager.net.train()
        # sample a batch of data from validation set
        images, labels = self.run_manager.run_config.valid_next_batch
        images, labels = images.to(self.run_manager.device), labels.to(self.run_manager.device)
        # sample nets and get their validation accuracy, latency, etc
        grad_buffer = []
        reward_buffer = []
        net_info_buffer = []
        for i in range(self.arch_search_config.batch_size):
            self.net.reset_binary_gates()  # random sample binary gates
            self.net.unused_modules_off()  # remove unused module for speedup
            # validate the sampled network
            with torch.no_grad():
                output = self.run_manager.net(images)
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            net_info = {'acc': acc1[0].item()}
            # get additional net info for calculating the reward
            if self.arch_search_config.target_hardware is None:
                pass
            elif self.arch_search_config.target_hardware == 'flops':
                net_info['flops'] = self.run_manager.net_flops()
            else:
                net_info[self.arch_search_config.target_hardware], _ = self.run_manager.net_latency(
                    l_type=self.arch_search_config.target_hardware, fast=fast
                )
            net_info_buffer.append(net_info)
            # calculate reward according to net_info
            reward = self.arch_search_config.calculate_reward(net_info)
            # loss term
            obj_term = 0
            for m in self.net.redundant_modules:
                if m.AP_path_alpha.grad is not None:
                    m.AP_path_alpha.grad.data.zero_()
                obj_term = obj_term + m.log_prob
            loss_term = -obj_term
            # backward
            loss_term.backward()
            # take out gradient dict
            grad_list = []
            for m in self.net.redundant_modules:
                grad_list.append(m.AP_path_alpha.grad.data.clone())
            grad_buffer.append(grad_list)
            reward_buffer.append(reward)
            # unused modules back
            self.net.unused_modules_back()
        # update baseline function
        avg_reward = sum(reward_buffer) / self.arch_search_config.batch_size
        if self.arch_search_config.baseline is None:
            self.arch_search_config.baseline = avg_reward
        else:
            self.arch_search_config.baseline += self.arch_search_config.baseline_decay_weight * \
                                                (avg_reward - self.arch_search_config.baseline)
        # assign gradients
        for idx, m in enumerate(self.net.redundant_modules):
            m.AP_path_alpha.grad.data.zero_()
            for j in range(self.arch_search_config.batch_size):
                m.AP_path_alpha.grad.data += (reward_buffer[j] - self.arch_search_config.baseline) * grad_buffer[j][idx]
            m.AP_path_alpha.grad.data /= self.arch_search_config.batch_size
        # apply gradients
        self.arch_optimizer.step()

        return reward_buffer, net_info_buffer

    def gradient_step(self):
        assert isinstance(self.arch_search_config, GradientArchSearchConfig)
        if self.arch_search_config.data_batch is None:
            self.run_manager.run_config.valid_loader.batch_sampler.batch_size = \
                self.run_manager.run_config.train_batch_size
        else:
            self.run_manager.run_config.valid_loader.batch_sampler.batch_size = self.arch_search_config.data_batch
        self.run_manager.run_config.valid_loader.batch_sampler.drop_last = True
        # switch to train mode
        self.run_manager.net.train()
        # Mix edge mode
        MixedEdge.MODE = self.arch_search_config.binary_mode
        time1 = time.time()  # time
        # sample a batch of data from validation set
        images, labels = self.run_manager.run_config.valid_next_batch
        images, labels = images.to(self.run_manager.device), labels.to(self.run_manager.device)
        time2 = time.time()  # time
        # compute output
        self.net.reset_binary_gates()  # random sample binary gates
        self.net.unused_modules_off()  # remove unused module for speedup
        output = self.run_manager.net(images)
        time3 = time.time()  # time
        # loss
        ce_loss = self.run_manager.criterion(output, labels)
        if self.arch_search_config.target_hardware is None:
            expected_value = None
        elif self.arch_search_config.target_hardware == 'mobile':
            expected_value = self.net.expected_latency(self.run_manager.latency_estimator)
        elif self.arch_search_config.target_hardware == 'flops':
            data_shape = [1] + list(self.run_manager.run_config.data_provider.data_shape)
            input_var = torch.zeros(data_shape, device=self.run_manager.device)
            expected_value = self.net.expected_flops(input_var)
        else:
            raise NotImplementedError
        loss = self.arch_search_config.add_regularization_loss(ce_loss, expected_value)
        # compute gradient and do SGD step
        self.run_manager.net.zero_grad()  # zero grads of weight_param, arch_param & binary_param
        loss.backward()
        # set architecture parameter gradients
        self.net.set_arch_param_grad()
        self.arch_optimizer.step()
        if MixedEdge.MODE == 'two':
            self.net.rescale_updated_arch_param()
        # back to normal mode
        self.net.unused_modules_back()
        MixedEdge.MODE = None
        time4 = time.time()  # time
        self.write_log(
            '(%.4f, %.4f, %.4f)' % (time2 - time1, time3 - time2, time4 - time3), 'gradient',
            should_print=False, end='\t'
        )
        return loss.data.item(), expected_value.item() if expected_value is not None else None
