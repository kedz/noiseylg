from ..types import register, PlumModule, HP, props, Variable
from torch.nn.functional import cross_entropy


@register("loss_functions.class_cross_entropy")
class ClassCrossEntropy(PlumModule):
    
    logits_field = HP(default="target_logits", type=props.STRING)
    labels_field = HP(default="targets", type=props.STRING)
    padding_index = HP(default=None, required=False, type=props.INTEGER)

    def __pluminit__(self):
        self._total_loss = 0
        self._num_items = 0

    def scalar_result(self):
        if self._num_items == 0:
            return 0
        else:
            return self._total_loss / self._num_items

    def reset(self):
        self._total_loss = 0
        self._num_items = 0

    def forward(self, forward_state, batch):

        logits = forward_state[self.logits_field]
        if isinstance(logits, Variable):
            logits = logits.permute_as_batch_sequence_features().data

        labels = batch[self.labels_field]
        if isinstance(labels, Variable):
            labels = labels.permute_as_batch_sequence_features()
            labels = labels.apply_sequence_mask(pad_value=self.padding_index)
            labels = labels.data

        if logits.dim() > 2:
            num_classes = logits.size(-1)
            logits = logits.contiguous().view(-1, num_classes)
            labels = labels.contiguous().view(-1)

        if self.padding_index is not None:
            num_items = labels.ne(self.padding_index).long().sum().item()
        else: 
            num_items = labels.numel()

        total_xent = cross_entropy(
            logits, labels, ignore_index=self.padding_index,
            reduction="sum")
        self._total_loss += total_xent.item()
        self._num_items += num_items

        return total_xent / num_items

    def compute(self):
        return {"cross_entropy": self.scalar_result()}
