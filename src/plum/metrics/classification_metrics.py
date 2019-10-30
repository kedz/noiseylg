from ..types import register, PlumModule, HP, P, props, LazyDict, Variable
from collections import OrderedDict
from sklearn.metrics import precision_recall_fscore_support


#TODO remove sklearn dependency, update a confusion matrix online.

@register("metrics.class_prf")
class ClassPRF(PlumModule):

    output_field = HP(default="output")
    targets_field = HP(default="targets")
    num_classes = HP(type=props.INTEGER)
    vocab = HP(required=False)

    def __pluminit__(self):
        self.reset()

    def reset(self):
        self._output = []
        self._targets = []
        self._cache = None

    def forward(self, forward_state, batch):
        output = self.apply_fields(self.output_field, forward_state)
        targets = self.apply_fields(self.targets_field, batch)
        if isinstance(output, Variable):
            self._output.extend(output.tensor.contiguous().view(-1).tolist())
        else:
            self._output.extend(output.contiguous().view(-1).tolist())
        if isinstance(targets, Variable):
            self._targets.extend(targets.tensor.contiguous().view(-1).tolist())
        else:
            self._targets.extend(targets.contiguous().view(-1).tolist())

    def _compute_metric(self):

        p, r, f, s = precision_recall_fscore_support(
            self._targets, self._output,
            labels=range(self.num_classes))

        result = OrderedDict()
        mean_f = 0
        mean_p = 0
        mean_r = 0
        n = sum([1 if x > 0 else 0 for x in s])

        for i in range(self.num_classes):
            if s[i] > 0:
                mean_f += float(f[i])
                mean_p += float(p[i])
                mean_r += float(r[i])
            label = str(i) if self.vocab is None else self.vocab[i]
            result[label] = OrderedDict(
                fscore=float(f[i]),
                precision=float(p[i]),
                recall=float(r[i]))
        mean_f /= n
        mean_p /= n
        mean_r /= n
        result["average"] = OrderedDict(fscore=mean_f, precision=mean_p,
                                        recall=mean_r)

        return result

    def compute(self):
        if self._cache is None:
            self._cache = self._compute_metric()
        return self._cache

    def pretty_result(self):
        result = self.compute()
        s = "AVG Prec.={precision:5.2f} Recall={recall:5.2f} F1={fscore:5.2f}"
        return s.format(**result["average"])

    def apply_fields(self, fields, obj):
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if hasattr(field, "__call__"):
                obj = field(obj)
            else:
                obj = obj[field]
        return obj
