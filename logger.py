"""Episode CSV + optional TensorBoard logging."""
import csv
import os


class Logger:
    def __init__(self, log_dir, tensorboard=True):
        os.makedirs(log_dir, exist_ok=True)
        self.csv_file = open(os.path.join(log_dir, 'episodes.csv'), 'a', newline='')
        self.writer = csv.writer(self.csv_file)
        if self.csv_file.tell() == 0:
            self.writer.writerow(['episode', 'iteration', 'num_obj', 'return', 'actions', 'success', 'epsilon'])
        self.tb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(os.path.join(log_dir, 'tb'))
            except Exception:  # tensorboard not installed: CSV only
                print('TensorBoard not available, logging to CSV only')

    def scalar(self, name, value, step):
        if self.tb is not None:
            self.tb.add_scalar(name, value, step)

    def episode(self, episode, iteration, num_obj, ret, actions, success, eps):
        self.writer.writerow([episode, iteration, num_obj, ret, actions, int(success), round(eps, 4)])
        self.csv_file.flush()
        self.scalar('episode/return', ret, iteration)
        self.scalar('episode/success', float(success), iteration)
        self.scalar('episode/actions', actions, iteration)
        self.scalar('episode/num_obj', num_obj, iteration)

    def close(self):
        self.csv_file.close()
        if self.tb is not None:
            self.tb.close()
