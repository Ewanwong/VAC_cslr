import torch
import torch.optim as optim


class Optimizer(object):
    def __init__(self, model, optim_dict):
        self.optim_dict = optim_dict

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=self.optim_dict['base_lr'],
            weight_decay=self.optim_dict['weight_decay']
        )
        self.scheduler = self.define_lr_scheduler(self.optim_dict['step'])

    def define_lr_scheduler(self, milestones):
        # divide by 5
        lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.2)
        return lr_scheduler

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def to(self, device):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)