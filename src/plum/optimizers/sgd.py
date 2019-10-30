from ..types import register, PlumObject, HP, props
import torch


@register("optimizer.sgd")
class SGD(PlumObject):

    lr = HP(type=props.POSITIVE)
    momentum = HP(default=0.0, type=props.NON_NEGATIVE)
    weight_decay = HP(default=0.0, type=props.NON_NEGATIVE)

    def __pluminit__(self):
        self._impl = None

    def zero_grad(self):
        self._impl.zero_grad()

    def step(self):
        self._impl.step()

    def setup_optimizer(self, trainer, verbose=False):
        self._impl = torch.optim.SGD(
            trainer.model.parameters(), lr=self.lr,
            momentum=self.momentum, weight_decay=self.weight_decay)
