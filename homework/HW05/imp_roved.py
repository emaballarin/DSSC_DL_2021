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


def _maskdrill(odict: Mask, okey: str, ikey: str) -> Set[int]:
    """Return the set contained in a Mask after double key lookup, empty if none"""
    if okey in odict.keys():
        if ikey in odict[okey].keys():
            return odict[okey][ikey]
        return set()
    return set()


def mask_size(maskin: Mask) -> int:
    """
    Count the number of elements inside a Mask object

    Parameters
    ----------
    maskin : Mask
        The Mask the number of elements of is required

    Returns
    -------
    int
        The number of elements in such mask
    """

    ret: int = 0
    for okey in maskin.keys():
        for ikey in maskin[okey].keys():
            ret += len(maskin[okey][ikey])
    return ret


def maskterialize(tsize, indexset: Set[int]) -> Tensor:
    """Materialize a Tensor mask from size and set of masked indices"""
    return th.ones(tsize).index_put_(
        values=th.tensor([0.0]), indices=(th.tensor([*indexset], dtype=th.long),)
    )


def magnitude_pruning(
    model,
    rate: realnum,
    restr_layers: Optional[Set[str]] = None,
    restr_parameters: Optional[Set[str]] = None,
    mask: Optional[Mask] = None,
) -> Mask:
    """
    Apply one iteration of Iterative Magnitude Pruning to a PyTorch model

    Parameters
    ----------
    model : Any valid PyTorch model
        The model to prune
    rate : Union[float, int]
        The pruning rate
    restr_layers : Optional[Set[str]], optional
        The number of the layers to which to restrict pruning. If None, no restriction is applied. Defaults to None.
    restr_parameters : Optional[Set[str]], optional
        The name of the tensor-parameters to which to restrict pruning. If None, no restriction is applied. Defaults to None.
    mask : Optional[Mask], optional
        A starting mask for the pruning process. Indexes denote already-pruned (or unprunable) parameters. Useful in case of repeated iterations of IMP. Defaults to None, equivalent to an empty mask.

    Returns
    -------
    Mask
        The pruning mask at the end of the IMP iteration.

    Raises
    ------
    ValueError
        In case a pruning rate incompatible with a ratio is passed.
    """

    # Validate signature
    if rate < 0 or rate > 1:
        raise ValueError(
            "Given pruning rate {} is negative or exceeds 1. Provide a valid rate!"
        )

    # Fill empty elements, if None
    if mask is None:
        mask: Mask = {}
    if restr_layers is None:
        restr_layers = []
    if restr_parameters is None:
        restr_parameters = []

    # First pass: look and select

    sel_par = []

    with th.no_grad():
        for name, param in model.named_parameters():

            layname = paramsplit(name, 1)[0]
            parname = paramsplit(name, 1)[1]

            if (not restr_layers or layname in restr_layers) and (
                not restr_parameters or parname in restr_parameters
            ):
                sel_par.append(
                    param.view(-1)[
                        list(
                            set(range(param.numel())).difference(
                                _maskdrill(mask, layname, parname)
                            )
                        )
                    ]
                    .clone()
                    .detach()
                    .abs()
                )

        flat_par = th.cat(sel_par, dim=0).sort()[0]

        # Pruning threshold
        thresh = flat_par[int(rate * flat_par.shape[0])]

        # Second pass: compare and prune

        for name, param in model.named_parameters():

            layname = paramsplit(name, 1)[0]
            parname = paramsplit(name, 1)[1]

            if (not restr_layers or layname in restr_layers) and (
                not restr_parameters or parname in restr_parameters
            ):
                flat_param = param.view(-1)
                tensormask = th.where(flat_param.abs() >= thresh, 1, 0)

                # Apply via the tensor-mask just created
                flat_param.data *= tensormask

                # Store the indexes of pruned elements
                if not layname in mask.keys():
                    mask[layname] = IdxSet()
                if not parname in mask[layname].keys():
                    mask[layname][parname] = set()
                mask[layname][parname].update(
                    set((tensormask == 0).nonzero().view(-1).tolist())
                )

    # Return line
    return mask
