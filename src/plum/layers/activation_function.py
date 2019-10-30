from ..types import register, PlumModule, HP, props, Variable
import torch


options = ["relu", "tanh", "sigmoid", "identity"]

@register("layers.activation_function")
class ActivationFunction(PlumModule):

    name = HP(type=props.Choice(options))
    inplace = HP(default=False, type=props.BOOLEAN)

    @property
    def func(self):
        return self._func

    def __pluminit__(self, name):
        if name == "relu":
            if self.inplace:
                self._func = torch.relu_
            else:
                self._func = torch.relu
        elif name == "tanh":
            self._func = torch.tanh
        elif name == "sigmoid":
            self._func = torch.sigmoid
        elif name == "identity":
            def identity_func(x): 
                return x
            self._func = identity_func
        else: 
            raise ValueError("name {} is not a valid option.".format(name))

    def forward(self, inputs):
        if isinstance(inputs, Variable):
            out = self._func(inputs.data)
            return inputs.new_with_meta(out)
        else:
            return self._func(inputs)
