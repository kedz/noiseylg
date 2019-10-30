from ..types import register, PlumModule, HP, SM


@register("layers.zip")
class Zip(PlumModule):

    modules = SM()
    aggregate = SM(required=False)

    def forward(self, inputs):

        assert len(inputs) == len(self.modules)

        outputs = []
        for input, module in zip(inputs, self.modules):
            outputs.append(module(input))

        if self.aggregate:
            return self.aggregate(outputs)
        else:
            return outputs
