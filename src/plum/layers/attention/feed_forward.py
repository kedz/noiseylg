from plum.types import register, PlumModule, P, HP, SM, props, LazyDict,\
    Variable
import torch
from ..identity import Identity


def _curry_composition(attention, value, value_net):
    def compose():
        x = value_net(value)
        c = attention.bmm(x).transpose(1, 0)
        return c
    return compose

@register("layers.attention.feed_forward")
class FeedForwardAttention(PlumModule):

    query_net = SM(default=Identity())
    key_net = SM(default=Identity())
    value_net = SM(default=Identity())
    hidden_size = HP(type=props.INTEGER)

    weight = P("hidden_size", tags=["weight", "fully_connected"])



     
    def forward(self, query, key, value=None):
        if value is None:
            value = key

        if isinstance(query, Variable):
            return self._variable_forward(query, key, value)
        else:
            return self._tensor_forward(query, key, value)
    
    def _variable_forward(self, query, key, value):
        key = self.key_net(key)
        query = self.query_net(query)

        key = key.permute_as_sequence_batch_features()
        query = query.permute_as_sequence_batch_features()

        assert key.dim() == query.dim() == 3
        # TODO use named tensors to allow aribitrary seq dims 
       
        with torch.no_grad():
            mask = ~torch.einsum("qbh,kbh->qkb", [(~query.mask).float(), 
                                                  (~key.mask).float()]).byte()
        
        query_uns = query.data.unsqueeze(query.length_dim + 1)
        key_uns = key.data.unsqueeze(query.length_dim)

        hidden = torch.tanh(key_uns + query_uns)
        scores = hidden.matmul(self.weight)

        scores = scores.masked_fill(mask, float("-inf"))

        attention = torch.softmax(scores.transpose(1,2), dim=2)
        attention = attention.masked_fill(attention != attention, 0.)

        comp = torch.einsum(
            "ijk,kjh->ijh",
            [attention, self.value_net(value).data])
        comp = Variable(comp, lengths=query.lengths, length_dim=0, 
                        batch_dim=1)

        return {"attention": attention, "output": comp}

    def _tensor_forward(self, query, key, value):

        key = self.key_net(key)
        query = self.query_net(query)

        key = key.unsqueeze(0)
        query = query.unsqueeze(1)

        hidden = torch.tanh(key + query)
        scores = hidden.matmul(self.weight)

        attention_batch_last = torch.softmax(scores, dim=1)
        attention_batch_second = attention_batch_last.transpose(1, 2)
        attention_batch_first = attention_batch_last.permute(2, 0, 1)

        value_batch_first = value.transpose(1, 0)
        
        result = LazyDict(attention=attention_batch_second)
        comp = _curry_composition(attention_batch_first, value_batch_first,
                                  self.value_net)
        result.lazy_set("output", comp)
                                
        return result
