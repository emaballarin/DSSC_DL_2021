#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ==============================================================================
#
# :: IMProved ::
#
# Improved tools for Iterative Magnitude Pruning and PyTorch model masking
# with minimal-memory impact, device invariance, O(1) amortized lookup and
# tensor-mask on-demand materialization.
#
# ==============================================================================

# IMPORTS:
from typing import Union, Optional, Tuple, Set, TypedDict
from torch import Tensor

import torch as th

# USELESS:
from scripts import architectures

# CUSTOM TYPES:

realnum = Union[float, int]


class IdxSet(TypedDict):
    """Typed dictionary for string-named sets of ints"""

    name: str
    val: Set[int]


class Mask(TypedDict):
    """Typed dictionary for string-named IdXSets"""

    name: str
    val: IdxSet


# FUNCTIONS:
def paramsplit(
    paramname: str, lfrom: Optional[int] = None, lto: Optional[int] = None
) -> Tuple[str]:
    """Extract a (layer name, parameter role) tuple form PyTorch model's named_parameters"""
    return tuple(paramname.split("."))[lfrom:lto]


def maskdrill(odict: Mask, okey: str, ikey: str) -> Set[int]:
    """Return the det contained in a Mask after double key lookup, empty of none"""
    if okey in odict.keys():
        if ikey in odict[okey].keys():
            return odict[okey][ikey]
        return set()
    return set()


def maskterialize(tsize, indexset: Set[int]) -> Tensor:
    """Materialize a Tensor mask from size and set of masked indices"""
    return th.ones(tsize).index_put_(
        values=th.tensor([0.0]), indices=(th.tensor([*indexset], dtype=th.long),)
    )


def magnitude_pruning(
    model,
    rate: realnum,
    restrict_layers: Optional[Set[str]] = None,
    restrict_parameters: Optional[Set[str]] = None,
    mask: Optional[Mask] = None,
) -> Mask:

    # Signature validation
    if rate < 0 or rate > 1:
        raise ValueError(
            "Given pruning rate {} is negative or exceeds 1. Provide a valid rate!"
        )
    if mask is None:
        mask: Mask = {}

    # Return line
    return mask


################################################################################
# TESTS ########################################################################
################################################################################
layers = [
    {"n_in": 1, "n_out": 2, "batchnorm": False},
    {"n_out": 3, "batchnorm": True},
    {"n_out": 4, "batchnorm": True},
]

net = architectures.MLPCustom(layers)

for name, param in net.named_parameters():
    if paramsplit(name, 1)[0] in {"1", "2"} and paramsplit(name, 1)[1] in {
        "bias",
        "fdsfdsf",
    }:
        print(param[0])
