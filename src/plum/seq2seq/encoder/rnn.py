from ...types import register, PlumModule, P, HP, SM, props
from ...layers import GRU


def init_state_dims(rnn_encoder):
    dims0 = 2 if rnn_encoder.bidirectional else 1
    dims0 *= rnn_encoder.num_layers
    dims2 = rnn_encoder.out_feats
    return [dims0, 1, dims2]

@register("seq2seq.encoder.rnn")
class RNNEncoder(PlumModule):

    input_net = SM()
    
    rnn_cell = HP(default="lstm", type=props.Choice(["rnn", "lstm", "gru"]))
    in_feats = HP(type=props.INTEGER)
    out_feats = HP(type=props.INTEGER)
    bidirectional = HP(default=False, type=props.BOOLEAN)
    num_layers = HP(default=1, type=props.INTEGER)
    dropout = HP(default=0.)
    learn_init_state = HP(default=False, type=props.BOOLEAN)

    init_hidden = P(init_state_dims, conditional="learn_init_state",
                    tags=["init_hidden_state"])


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

    def init_state(self, batch_size=1):
        if self.learn_init_state:
            if self.rnn_cell == "gru" or self.rnn_cell == "rnn":
                return self.init_hidden.repeat(1, batch_size, 1)
            else:
                return (
                    self.init_hidden.repeat(1, batch_size, 1),
                    self.init_output.repeat(1, batch_size, 1),
                )
        else:
            return None

    def forward(self, inputs):

        #batch_size = inputs.size(1)
        rnn_input = self.input_net(inputs)
        batch_size = rnn_input.size(1)
        output, state = self.rnn(rnn_input, self.init_state(batch_size))

        return {"output": output, "state": state}
