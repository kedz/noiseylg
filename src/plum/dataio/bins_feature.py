from plum.types import register, PlumObject, HP


@register("dataio.pipeline.threshold_feature")
class ThresholdFeature(PlumObject):

    thresholds = HP()

    def __len__(self):
        return len(self.thresholds) + 1

    def __call__(self, value):
        if not isinstance(value, (int, float)):
            raise Exception("Expecting numerical values, int or float.")
        # this should be a binary search but I'm tired.
        bin = 0
        
        while bin != len(self.thresholds) and value > self.thresholds[bin]:
            bin += 1

#        if bin == 0:
#            print(value, self.thresholds[bin])
#        elif bin == len(self.thresholds):
#            print(self.thresholds[bin-1], vaulue)
#        else:
#            print(self.thresholds[bin-1], value, self.thresholds[bin])
        
        return bin


