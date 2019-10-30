from ..types import register, PlumObject, HP
import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from math import log10, ceil


@register("optimizer.sgd_1cycle")
class SGD_1Cycle(PlumObject):
    
    min_lr = HP(default=1e-8)
    max_lr = HP(default=1e1)
    max_iters = HP(default=1000)
    weight_decays = HP(default=[0])
    beta = HP(default=9.8e-1)
    max_momentum = HP(default=0.0)
    min_momentum = HP(default=0.0)
    up_percent = HP(default=0.45)
    down_percent = HP(default=0.45)
    
    def __pluminit__(self):
        self._optim = None
        self._current_iter = 0

    def zero_grad(self):
        self.optim.zero_grad()

    def step(self):
        lr = self._lrs[self._current_iter]
        momentum = self._moms[self._current_iter]
        self.optim.param_groups[0]["lr"] = lr
        self.optim.param_groups[0]["momentum"] = momentum
        self.optim.step()
        self._current_iter += 1

    @property
    def update_factor(self):
        return (self.max_lr / self.min_lr) ** (1 / self.max_iters)

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, params):
        self._parameters = params

    @property
    def optim(self):
        return self._optim

    @optim.setter
    def optim(self, val):
        self._optim = val

    def setup_optimizer(self, trainer, verbose=False):

        best_settings = []
        trainer.model.train()
        original_params = {n: p.clone() 
                           for n, p in trainer.model.named_parameters()}
        
        for round, weight_decay in enumerate(self.weight_decays, 1):
            for n, p in trainer.model.named_parameters():
                p.data.copy_(original_params[n])
            self.optim = torch.optim.SGD(trainer.model.parameters(), 
                                         lr=self.min_lr, 
                                         weight_decay=weight_decay,
                                         momentum=self.max_momentum)
            avg_loss = 0
            losses = []
            log_lrs = []


            lr = self.min_lr
            step = 0
            while step < self.max_iters:
                for batch in trainer.train_batches:
                    step += 1
                    if verbose:
                        print("lr finder: {}/{}".format(step, self.max_iters),
                            end="\r" if step < self.max_iters else "\n",
                            flush=True)

                    self.optim.zero_grad()

                    loss = trainer.loss_function(trainer.model(batch), batch)
                    loss.backward()
                    self.optim.step()
                    avg_loss = (
                        self.beta * loss.item() + (1 - self.beta) * avg_loss
                    )
                    smoothed_loss = avg_loss / (1 - self.beta ** step)
                    losses.append(smoothed_loss)
                    log_lrs.append(log10(lr))

                    lr *= self.update_factor
                    for pg in self.optim.param_groups:
                        pg["lr"] = lr
                    if step == self.max_iters:
                        break
    
            best_lr = 10 ** log_lrs[np.argmin(losses)]
            best_loss = np.min(losses)
            best_settings.append({"lr": best_lr, "loss": best_loss, 
                                  "weight_decay": weight_decay})

            plt.plot(log_lrs, losses, label="L2 pen={}".format(weight_decay))
        plt.legend()
        plt.xlabel("lr (log base 10)")
        plt.ylabel("loss")
        plt.savefig("lr_search.pdf")

        best_settings.sort(key=lambda x: x["loss"])
        max_lr = best_settings[0]["lr"]
        min_lr = max_lr / 10
        weight_decay = best_settings[0]["weight_decay"]

        total_iters = trainer.max_epochs * len(trainer.train_batches)
        up_duration = ceil(total_iters * self.up_percent)
        down_duration = ceil(total_iters * self.down_percent)
        anneal_duration = max(0, total_iters - up_duration - down_duration)

        start_phase_lrs = np.linspace(min_lr, max_lr, num=up_duration)
        stop_phase_lrs = np.linspace(max_lr, min_lr, num=down_duration)
        
        lrs = np.hstack([start_phase_lrs, stop_phase_lrs])
        if anneal_duration > 0:
            lrs = np.hstack(
                [lrs, np.linspace(min_lr, min_lr / 100, num=anneal_duration)])
        if total_iters != lrs.shape[0]:
            raise Exception("lr steps doesn't match total iterations.")

        if self.max_momentum > 0:
            up_phase_moms = np.linspace(
                self.max_momentum, self.min_momentum, num=up_duration)
            down_phase_moms = np.linspace(
                self.min_momentum, self.max_momentum, num=down_duration)
            anneal_phase_moms = np.array([self.max_momentum] * anneal_duration)
            moms = np.hstack([up_phase_moms, down_phase_moms, 
                              anneal_phase_moms])
        else:
            moms = np.array([0] * total_iters)

        plt.figure()
        plt.plot(lrs)
        plt.savefig("lr_schedule.pdf")

        plt.figure()
        plt.plot(moms)
        plt.savefig("momentum_schedule.pdf")

        if verbose:
            print(
                "1Cycle learning rate lr=({:f},{:f})".format(min_lr, max_lr) \
                        + " momentum=({:f},{:f}) weight_decay={:f}".format(
                    self.min_momentum, self.max_momentum, weight_decay))

        self._lrs = lrs
        self._moms = moms
        for n, p in trainer.model.named_parameters():
            p.data.copy_(original_params[n])
        self.optim = torch.optim.SGD(trainer.model.parameters(),
                                     lr=lrs[0], momentum=moms[0],
                                     weight_decay=weight_decay)
