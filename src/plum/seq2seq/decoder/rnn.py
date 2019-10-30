from ...types import register, PlumModule, P, HP, SM, props, Variable
from ...layers import GRU, Identity
import torch
import plum


@register("seq2seq.decoder.rnn")
class RNNDecoder(PlumModule):

    input_net = SM()

    rnn_cell = HP(default="lstm", type=props.Choice(["rnn", "lstm", "gru"]))
    in_feats = HP(type=props.INTEGER)
    out_feats = HP(type=props.INTEGER)
    bidirectional = HP(default=False, type=props.BOOLEAN)
    num_layers = HP(default=1, type=props.INTEGER)
    dropout = HP(default=0.0)

    attention_net = SM()
    pre_output_net = SM(default=Identity())
    predictor_net = SM()

    def __pluminit__(self, in_feats, out_feats, bidirectional, num_layers,
                     dropout):

        if self.rnn_cell == "rnn":
            raise Exception()
            rnn_cons = RNN
        elif self.rnn_cell == "lstm":
            raise Exception()
            rnn_cons = LSTM
        elif self.rnn_cell == "gru":
            rnn_cons = GRU
        else:
            raise ValueError(
                "rnn_cell has illegal value: {}".format(self.rnn_cell))
        
        self.rnn = rnn_cons(in_feats=in_feats, out_feats=out_feats, 
                            bidirectional=bidirectional,
                            num_layers=num_layers,
                            dropout=dropout)

    def _concat_controls(self, rnn_input, controls):
        assert controls.length_dim == rnn_input.length_dim
        assert controls.batch_dim == rnn_input.batch_dim
        assert controls.length_dim == 0
        seq_size = rnn_input.size(rnn_input.length_dim)            
        ctrl_data = controls.data.repeat(seq_size, 1, 1)
        new_data = torch.cat([rnn_input.data, ctrl_data], dim=2)
        return rnn_input.new_with_meta(new_data)

    def forward(self, inputs, encoder_state, prev_decoder_state=None,
                controls=None):

        rnn_input = self.input_net(inputs)
        if prev_decoder_state is None:
            prev_decoder_state = encoder_state["state"]
        
        if controls is not None:
            rnn_input = self._concat_controls(rnn_input, controls)

        if isinstance(rnn_input, Variable):
            # dont use length aware rnn here, avoids unpacking and need for 
            # in order inputs 
            rnn_output, rnn_state = self.rnn(rnn_input.data, 
                                             prev_decoder_state)
            rnn_output = rnn_input.new_with_meta(rnn_output)
        else:
            rnn_output, rnn_state = self.rnn(rnn_input.data, 
                                             prev_decoder_state)
        
        attention = self.attention_net(rnn_output, encoder_state["output"])
        if attention["output"] is not None:
            hidden_state = plum.cat([rnn_output, attention["output"]], dim=2)
        else:
            hidden_state = rnn_output

        pre_output = self.pre_output_net(hidden_state)
        output = self.predictor_net(pre_output)

        output["decoder_state"] = rnn_state
        return output

    def next_state(self, prev_state, search_context, controls=None):
        decoder_state = prev_state["decoder_state"]
        return self.forward(
            prev_state["output"], 
            {"output": search_context["encoder_output"]},
            prev_decoder_state=decoder_state,
            controls=controls)
