from pathlib import Path


class NonNegative:

    def __call__(self, value):
        return value >= 0
        
NON_NEGATIVE = NonNegative()

class Positive:
    def __call__(self, value):
        return value > 0

POSITIVE = Positive()

class Integer:
    def __call__(self, value):
        return isinstance(value, int) or value == int(value)

INTEGER = Integer()

class Boolean:
    def __call__(self, value):
        return isinstance(value, bool)

BOOLEAN = Boolean()

class Real:
    def __call__(self, value):
        # Check for nans.
        if value != value:
            return False
        # Check general float-hood.
        if isinstance(value, float):
            return True
        try:    
            return value == float(value)
        except ValueError:
            return False

REAL = Real()

class Interval:
    def __init__(self, lb, ub, lb_closed=True, ub_closed=True):
        if lb > ub:
            raise ValueError("Lower bound cannot be greater than upper bound.")
        if lb == ub and not (lb_closed and ub_closed):
            raise ValueError(
                "Lower bound cannot equal upper bound in an open interval.")
        self.lb = lb
        self.ub = ub
        self.lb_closed = lb_closed
        self.ub_closed = ub_closed

    def __call__(self, value):
        gt_lb = value > self.lb or (self.lb_closed and value == self.lb)
        lt_ub = value < self.ub or (self.ub_closed and value == self.ub)
        return gt_lb and lt_ub

class Choice:
    def __init__(self, options):

        # cast to set in case iterator is passed.
        options = set(options)
        if len(options) == 0:
            raise ValueError("Must provide at least one option.")
        self.options = options
    
    def __call__(self, value):
        return value in self.options

class ExistingPath:
    
    def __call__(self, value):
        return Path(value).exists()        
        
EXISTING_PATH = ExistingPath()

class String:

    def __call__(self, value):
        return isinstance(value, str)

STRING = String()
