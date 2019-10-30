from ..types import register, PlumModule, SM


@register("metrics.metric_dict")
class MetricDict(PlumModule):
    
    metrics = SM()

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

    def forward(self, forward_state, batch):
        for metric in self.metrics.values():
            metric(forward_state, batch)

    def compute(self):
        return {name: metric.compute()
                for name, metric in self.metrics.items()}

    def pretty_result(self):
        buffer = []
        for name, metric in self.metrics.items():
            buffer.append(name)
            buffer.append(metric.pretty_result())
            
        return "\n".join(buffer)

