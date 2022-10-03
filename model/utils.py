import copy

from torch.nn import ModuleList


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])
