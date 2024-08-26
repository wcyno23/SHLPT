import os
import time



class Visualizer():

    def __init__(self, opt):
        self.opt = opt  # cache the option

        self.train_log_name = os.path.join(opt.checkpoints_dir, opt.data_name, opt.name, 'train_log.txt')
        with open(self.train_log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

        self.log_name = os.path.join(opt.checkpoints_dir, opt.data_name, opt.name, 'valid_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def print_valid_losses(self, epoch, losses, t_comp):
        message = '(epoch: %d, time: %.3f) ' % (epoch, t_comp)
        for k, v in losses.items():
            message += '%s: %.6f ' % (k, v)
        print("Valid: ", message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    def print_current_losses(self, epoch, iters, task, losses, t_comp):
        message = '(epoch: %d, process: %d/%d, time: %.3f) ' % (epoch, iters, task, t_comp)
        for k, v in losses.items():
            message += '%s: %.6f ' % (k, v)

        print("Train: ", message)  # print the message
        with open(self.train_log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message