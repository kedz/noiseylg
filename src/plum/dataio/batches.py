from ..types import register, PlumObject, HP, props
from torch.utils.data import DataLoader
from math import ceil


@register("dataio.batches")
class Batches(PlumObject):

    dataset = HP()
    batch_size = HP(type=props.INTEGER)
    shuffle = HP(default=True, type=props.BOOLEAN)
    num_workers = HP(default=1, type=props.INTEGER)
    pipelines = HP()
    collate_funcs = HP()

    sort = HP(default=False, type=props.BOOLEAN)
    sort_key = HP(required=False)
    sort_descending = HP(default=True, type=props.BOOLEAN)

    def __pluminit__(self):
        self._gpu = -1

    @property
    def gpu(self):
        return self._gpu

    @gpu.setter
    def gpu(self, new_val):
        self._gpu = new_val

    def _apply_transforms(self, item, transforms):
        if not isinstance(transforms, (list, tuple)):
            transforms = [transforms]

        for transform in transforms:
            if hasattr(transform, "__call__"):
                item = transform(item)
            else:
                item = item[transform]
        return item

    def sort_batch(self, batch):

        if self.sort_key is None:
            keys = [len(item) for item in batch]
        else:
            keys = [self._apply_transforms(item, self.sort_key)
                    for item in batch]

        indices = list(range(len(batch)))
        indices.sort(key=lambda x: keys[x], reverse=self.sort_descending)
        return [batch[i] for i in indices]

    def _collate_fn(self, batch):

        if self.sort:
            batch = self.sort_batch(batch)
        
        output = {pipeline: list() for pipeline in self.pipelines}

        for item in batch:
            for name, transforms in self.pipelines.items():
                output[name].append(self._apply_transforms(item, transforms))

        for name, transforms in self.collate_funcs.items():
            if name in output:
                data = output[name]
            else:
                data = batch
            output[name] = self._apply_transforms(data, transforms)

        return output

    def __iter__(self):
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn)

        for batch in dataloader:
            if self.gpu > -1:
                batch = self.batch2gpu(batch)
            
            yield batch

    def __len__(self):
        return ceil(len(self.dataset) / self.batch_size)

    def batch2gpu(self, batch):
        if isinstance(batch, dict):
            for key, value in batch.items():
                batch[key] = self.batch2gpu(value)
            return batch
        elif isinstance(batch, list):
            return [self.batch2gpu(item) for item in batch] 
        elif isinstance(batch, tuple):
            return tuple([self.batch2gpu(item) for item in batch])
        elif hasattr(batch, "cuda"):
            return batch.cuda(self.gpu)
        else:
            return batch
