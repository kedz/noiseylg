import torch


class Variable:

    def __init__(self, data, lengths=None, length_dim=None,
                 batch_dim=None, pad_value=0):

        self._data = data
        self._lengths = lengths
        self._length_dim = length_dim
        self._mask = None
        self._batch_dim = batch_dim
        self._pad_value = pad_value

        feature_dims = [i for i in range(self.dim())
                        if i not in [length_dim, batch_dim]]
        self._feature_dims = tuple(feature_dims)

    # TODO deprectate data in favor of tensor. pytorch tensors have data.
    # and plum variables have data. This is confusing. Instead we are moving to
    # plum.types.Variable -> tensor (torch.Tensor) -> data (torch.Storage) 
    @property
    def data(self):
        return self._data

    @property
    def tensor(self):
        return self._data

    @property
    def lengths(self):
        return self._lengths

    @property
    def length_dim(self):
        return self._length_dim

    def new_with_meta(self, tensor, lengths=None):
        return Variable(
            tensor, 
            lengths=lengths if lengths is not None else self.lengths,
            length_dim=self.length_dim,
            batch_dim=self.batch_dim,
            pad_value=self.pad_value)  

    @property
    def batch_dim(self):
        return self._batch_dim

    @property
    def batch_size(self):
        return self.data.size(self.batch_dim)
    
    def cuda(self, device):
        new = self.new_with_meta(self.data.cuda(device))
        new._lengths = new.lengths.cuda(device)
        return new

    def size(self, *args, **kwargs):
        return self._data.size(*args, **kwargs)

    def dim(self):
        return self._data.dim()

    @property
    def mask(self):
        if self._mask is not None:
            return self._mask
        ldim = self.length_dim
        max_len = self.data.size(ldim)
        
        dims = [1] * self.dim()
        dims[ldim] = max_len

        len_dims = [1] * self.dim()
        len_dims[self.batch_dim] = self.lengths.size(0)

        with torch.no_grad():
            x = torch.arange(max_len, device=self.data.device).view(*dims)
            bs = self.data.size(self.batch_dim)
            repeat_args = [1] * self.dim()
            repeat_args[self.batch_dim] = bs
            
            x = x.repeat(*repeat_args)
  
            self._mask = x >= self.lengths.view(*len_dims)
        return self._mask

    def masked_fill(self, mask, fill_value):
        data = self.data.masked_fill(mask, fill_value)
        return self.new_with_meta(data)

    def apply_sequence_mask(self, pad_value=None):
        new_data = self.data.masked_fill(
            self.mask,
            pad_value if pad_value is not None else self.pad_value)
        return self.new_with_meta(new_data)

    def apply_sequence_mask_(self, pad_value=None):
        self.data.data.masked_fill_(
            self.mask,
            pad_value if pad_value is not None else self.pad_value)
        return self

    def new(self, *args, **kwargs):
        return self.data.new(*args, **kwargs)

    def __repr__(self):
        return repr(self.data)

    @property
    def pad_value(self):
        return self._pad_value

    def pad_batch_dim(self, pad_size):
        if pad_size == 0:
            new_data = self.data.clone()
            new_lengths = self.lengths
        else:
            dims = list(self.data.size())
            dims[self.batch_dim] = pad_size
            pad = self.data.new(*dims).fill_(self.pad_value)
            new_data = torch.cat([self.data, pad], dim=self.batch_dim)
            new_lengths = torch.cat([self.lengths, 
                                     self.lengths.new([0] * pad_size)])
        return Variable(new_data, lengths=new_lengths, 
                        batch_dim=self.batch_dim,
                        length_dim=self.length_dim, pad_value=self.pad_value) 

    def pad_length_dim(self, pad_size):
        if pad_size == 0:
            new_data = self.data.clone()
        else:
            dims = list(self.data.size())
            dims[self.length_dim] = pad_size
            pad = self.data.new(*dims).fill_(self.pad_value)
            new_data = torch.cat([self.data, pad], dim=self.length_dim)
        return self.new_with_meta(new_data)

    def repeat_batch_dim(self, size):
        rsz = [1] * (self.dim() + 1)
        rsz[self.batch_dim + 1] = size
        new_dims = list(self.size())
        new_dims[self.batch_dim] *= size
        new_data = self.data.unsqueeze(self.batch_dim + 1)\
            .contiguous().repeat(*rsz).view(*new_dims)
        new_lengths = self.lengths.view(-1, 1).repeat(1, size).view(-1)

        return Variable(new_data, lengths=new_lengths,
                        length_dim=self.length_dim,
                        batch_dim=self.batch_dim,
                        pad_value=self.pad_value)

    def softmax(self, dim=None):
        new_data = self.data.softmax(dim)
        return self.new_with_meta(new_data)

    def log_softmax(self, dim=None):
        new_data = self.data.log_softmax(dim)
        return self.new_with_meta(new_data)

    def normal_(self, *args, **kwargs):
        self.data.normal_(*args, **kwargs)
        return self

    def index_select(self, dim, index):
        new_data = self.data.index_select(dim, index)
        if dim != self.length_dim:
            return self.new_with_meta(new_data)
        else:
            if (index.view(-1, 1) >= self.lengths.view(1, -1)).any():
                raise RuntimeError(
                    "Indexing sequence dim beyond sequence length.")
            new_lengths = self.lengths.new([index.size(0)] * self.batch_size)
            return self.new_with_meta(new_data, lengths=new_lengths)

    def argmax(self, dim=None):
        # TODO make sure this respects mask
        if dim is None:
            return self.data.argmax()
        elif dim != self.length_dim and dim != self.batch_dim:
            new_data = self.data.argmax(dim)
            return self.new_with_meta(new_data)
        elif dim == self.length_dim:
            return self.data.argmax(dim)
        else:
            raise RuntimeError("Cannot argmax over batch dim.")

    def clone(self):
        return self.new_with_meta(self.data.clone(), self.lengths.clone())

    @property
    def feature_dims(self):
        return self._feature_dims

    def permute_as_batch_features_sequence(self):
        # Permute variable so it is batch x features x sequence

        dims = (self.batch_dim,) + self.feature_dims + (self.length_dim,)
        new_data = self.data.permute(*dims)
        return Variable(new_data, lengths=self.lengths, 
                        length_dim=len(dims) - 1, batch_dim=0,
                        pad_value=self.pad_value)

    def permute_as_batch_sequence_features(self):
        # Permute variable so it is batch x sequence x features

        dims = (self.batch_dim, self.length_dim) + self.feature_dims
        new_data = self.data.permute(*dims)
        return Variable(new_data, lengths=self.lengths, 
                        length_dim=1, batch_dim=0,
                        pad_value=self.pad_value)

    def permute_as_sequence_batch_features(self):
        # Permute variable so it is batch x sequence x features

        dims = (self.length_dim, self.batch_dim) + self.feature_dims
        new_data = self.data.permute(*dims)
        return Variable(new_data, lengths=self.lengths, 
                        length_dim=0, batch_dim=1,
                        pad_value=self.pad_value)

    def max(self, dim):

        if dim == self.batch_dim or dim == self.length_dim:
            raise Exception("Can't max batch or length dim.")
        maxes, argmaxes = self.data.max(dim)
        return self.new_with_meta(maxes), self.new_with_meta(argmaxes)
        

    def permute(self, *dims):
        
        new_length_dim = dims.index(self.length_dim)
        new_batch_dim = dims.index(self.batch_dim)
        
        new_data = self.data.permute(*dims)
        return Variable(
            new_data,
            lengths=self.lengths,
            batch_dim=new_batch_dim,
            length_dim=new_length_dim,
            pad_value=self.pad_value)

    def transpose(self, dim1, dim2):
        if dim1 == self.batch_dim: 
            new_batch_dim = dim2
        elif dim2 == self.batch_dim:
            new_batch_dim = dim1
        else:
            new_batch_dim = self.batch_dim
        if dim1 == self.length_dim: 
            new_length_dim = dim2
        elif dim2 == self.length_dim:
            new_length_dim = dim1
        else:
            new_length_dim = self.length_dim

        new_data = self.data.transpose(dim1, dim2)

        return Variable(
            new_data,
            lengths=self.lengths,
            batch_dim=new_batch_dim,
            length_dim=new_length_dim,
            pad_value=self.pad_value)

    def reduce_sequence(self, reduction="sum"):
        tensor_agg = self.apply_sequence_mask(0.).tensor.sum(self.length_dim)
        if reduction == "sum":
            return tensor_agg
        elif reduction == "mean":
            dims = [1] * self.dim()
            dims[self.batch_dim] = self.batch_size
            dims.pop(self.length_dim)
            return tensor_agg.float() / self.lengths.float().view(*dims)
        else:
            raise ValueError("reduction must be sum or mean")
