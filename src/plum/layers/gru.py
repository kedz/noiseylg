from ..types import register, PlumModule, P, HP, SM, props, Variable
import torch
import torch.nn as nn
from .functional import dropout
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@register("layers.gru")
class GRU(PlumModule):

    in_feats = HP(type=props.INTEGER)
    out_feats = HP(type=props.INTEGER)
    bidirectional = HP(default=False, type=props.BOOLEAN)
    num_layers = HP(default=1, type=props.INTEGER)
    dropout = HP()

    def __pluminit__(self, in_feats, out_feats, bidirectional, num_layers,
                     dropout):

        self._net = nn.GRU(
            in_feats, 
            out_feats, 
            bidirectional=bidirectional,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.)

        self._net._parameter_tags = {}
        for name, _ in self.named_parameters():
            tags = ["recurrent", "gru"]
            if "weight" in name:
                tags.append("weight")
                if "ih" in name:
                    tags.append("input_recurrence")
                else:
                    tags.append("hidden_recurrence")
            else:
                tags.append("bias")
                if "ih" in name:
                    tags.append("input_recurrence")
                else:
                    tags.append("hidden_recurrence")
            if "reverse" in name:
                tags.append("reverse")
            else:
                tags.append("forward")

            self._net._parameter_tags[name[5:]] = tuple(tags)

    def _variable_forward(self, inputs, prev_state):
        #if not torch.all(inputs.lengths[:-1] >= inputs.lengths[1:]):
        #    print(inputs.lengths)
        #    raise Exception(
        #        "Variable length inputs must be sorted in descending order.")
        packed_inputs = pack_padded_sequence(inputs.data, inputs.lengths, 
                                             batch_first=False,
                                             enforce_sorted=True)
        packed_outputs, state = self._net(packed_inputs, prev_state)
        outputs, _ = pad_packed_sequence(packed_outputs)

        return inputs.new_with_meta(outputs), state

    def forward(self, inputs, prev_state=None):
        if isinstance(inputs, Variable):
            output, state = self._variable_forward(inputs, prev_state)
        else:
            output, state = self._net(inputs, prev_state)
        output = dropout(output, p=self.dropout, training=self.training)
        state = dropout(state, p=self.dropout, training=self.training)

        return output, state
