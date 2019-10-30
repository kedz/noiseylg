from ..types import register, PlumModule, HP, SM
import torch


class PlumModel(PlumModule):

    initializers = HP()

    def initialize_parameters(self, verbose=False):
        
        initialized = set()
        name2param = {n: p for n, p in self.named_parameters()}
        
        for match_str, initializer in self.initializers.items():
            match_items = match_str.split(" ")   
            for name in name2param.keys():
                tags = self.parameter_tags(name)
                matches = []
                for item in match_items:
                    if item in ["&", "|"]:
                        matches.append(item)
                    else:
                        matches.append(str(item in tags))
                match = eval(" ".join(matches))
                if match:
                    if name in initialized:
                        from warnings import warn
                        warn("Parameter {} already initialized!")
                    if verbose:
                        print("{} <= {}".format(name, initializer.plum_id))
                    initializer(name2param[name])
                    initialized.add(name)

        for name in name2param.keys():
            if name not in initialized:
                if name2param[name].dim() >= 2:
                    if verbose:
                        print("{} <= (default) Xavier normal.".format(name))
                    torch.nn.init.xavier_normal_(name2param[name])
                else:
                    if verbose:
                        print("{} <= (default) standard normal.".format(name))
                    torch.nn.init.normal_(name2param[name])
        if verbose:
            print()
        return self
