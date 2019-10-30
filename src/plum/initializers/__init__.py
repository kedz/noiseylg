from plum.types import register, PlumObject, HP, props
import torch


@register("initializers.normal")
class Normal(PlumObject):
    
    mean = HP(type=props.REAL)
    std = HP(type=props.POSITIVE)

    def __call__(self, tensor):
        torch.nn.init.normal_(tensor, mean=self.mean, std=self.std)

@register("initializers.xavier_normal")
class XavierNormal(PlumObject):
    def __call__(self, tensor):
        torch.nn.init.xavier_normal_(tensor)

@register("initializers.constant")
class Constant(PlumObject):
    
    value = HP(type=props.REAL)

    def __call__(self, tensor):
        torch.nn.init.constant_(tensor, self.value)

