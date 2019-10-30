import torch
from .types import Variable


def cat(tensors, dim=0, out=None):
    if isinstance(tensors[0], Variable) and tensors[0].length_dim is not None:

        if dim == tensors[0].length_dim:
            raise Exception("Cannot concat along length dim: {}".format(dim))

        if dim == tensors[0].batch_dim:
            raise Exception("Cannot concat along batch dim: {}".format(dim))
        
        len0 = tensors[0].lengths
        for tr in tensors[1:]:
            if not torch.all(tr.lengths == len0):
                raise Exception((
                    "Variables can only be concatenated if they "
                    "have the same length!"))
        data = torch.cat([t.data for t in tensors], dim=dim, out=out)
        return tensors[0].new_with_meta(data)
    else:
        return torch.cat(tensors, dim=dim, out=out)    
