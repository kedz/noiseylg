from plum.types import Variable
import torch.nn.functional as F


def embedding(input, weight, **kwargs):
    if isinstance(input, Variable):
        output = F.embedding(input.data, weight, **kwargs)
        return input.new_with_meta(output)
    else:
        return F.embedding(input, weight, **kwargs)

def linear(input, weight, bias=None, in_dim=None):

    last_dim = input.dim() - 1

    if in_dim is None:
        if isinstance(input, Variable) and last_dim not in input.feature_dims:
            input = input.permute_as_batch_sequence_features()
        in_dim = last_dim
    
    if in_dim != last_dim:
        input = input.transpose(in_dim, last_dim)

    if isinstance(input, Variable):
        if input.length_dim == input.dim() - 1:
            raise Exception("Cannot contract length dimension!")
        output = F.linear(input.data, weight, bias)
        return input.new_with_meta(output)
    else:
        return F.linear(input, weight, bias)

def dropout(input, **kwargs):
    # TODO possibly modift Variable object inplace if specified.
    if isinstance(input, Variable):
        output = F.dropout(input.data, **kwargs)
        return input.new_with_meta(output)
    else:
        return F.dropout(input, **kwargs)

def seq_max_pool1d(input, kernel_size=None):
    assert input.dim() == 3
    if kernel_size is None:
        kernel_size = input.size(input.length_dim)
    input = input.permute_as_batch_features_sequence()\
        .apply_sequence_mask_(pad_value=float("-inf"))
    return F.max_pool1d(input.data, kernel_size=kernel_size)

def seq_conv1d(input, weight, **kwargs):
    
    assert input.dim() == 3
    input = input.permute_as_batch_features_sequence()
    output_data = F.conv1d(input.data, weight, **kwargs)

    kernel_width = weight.size(2)
    lengths_data = input.lengths + (
        2 * kwargs.get("padding", 0) - (kernel_width - 1)
    )
    return input.new_with_meta(output_data, lengths_data)

