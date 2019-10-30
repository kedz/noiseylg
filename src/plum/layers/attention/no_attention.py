from ...types import register, PlumModule


@register("layers.attention.none")
class NoAttention(PlumModule):
    def forward(self, query, key, value=None, prev_state=None):
        return {"output": query, "attention": None}
