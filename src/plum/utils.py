def resolve_getters(fields, item, default=None):
    if fields is None:
        return default
    if not isinstance(fields, (list, tuple)):
        fields = [fields]
    for field in fields:
        if hasattr(field, "__call__"):
            item = field(item)
        else:
            item = item[field]
    return item
