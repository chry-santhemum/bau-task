# Copied from trasformer_lens.patching
from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union, overload
import itertools
import pandas as pd
import torch
from jaxtyping import Float, Int
from tqdm.auto import tqdm
from typing_extensions import Literal

import transformer_lens.utils as utils
from transformer_lens.ActivationCache import ActivationCache
from transformer_lens.HookedTransformer import HookedTransformer

Logits = torch.Tensor
AxisNames = Literal["layer", "pos", "head_index", "head", "src_pos", "dest_pos"]

CorruptedActivation = torch.Tensor
PatchedActivation = torch.Tensor


# %%

def make_df_from_ranges(
    column_max_ranges: Sequence[int], column_names: Sequence[str]
) -> pd.DataFrame:
    """
    Takes in a list of column names and max ranges for each column, and returns a dataframe with the cartesian product of the range for each column (ie iterating through all combinations from zero to column_max_range - 1, in order, incrementing the final column first)
    """
    rows = list(itertools.product(*[range(axis_max_range) for axis_max_range in column_max_ranges]))
    df = pd.DataFrame(rows, columns=column_names)
    return df


def generic_activation_patch(
    model: HookedTransformer,
    corrupted_tokens: Int[torch.Tensor, "batch pos"],
    clean_cache: ActivationCache,
    patching_metric: Callable[[Float[torch.Tensor, "batch pos d_vocab"]], Float[torch.Tensor, ""]],
    patch_setter: Callable[
        [CorruptedActivation, Sequence[int], ActivationCache], PatchedActivation
    ],
    activation_name: str,
    index_axis_names: Optional[Sequence[AxisNames]] = None,
    index_df: Optional[pd.DataFrame] = None,
    return_index_df: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, pd.DataFrame]]:

    if index_df is None:
        assert index_axis_names is not None

        # Get the max range for all possible axes
        max_axis_range = {
            "layer": model.cfg.n_layers,
            "pos": corrupted_tokens.shape[-1],
            "head_index": model.cfg.n_heads,
        }
        max_axis_range["src_pos"] = max_axis_range["pos"]
        max_axis_range["dest_pos"] = max_axis_range["pos"]
        max_axis_range["head"] = max_axis_range["head_index"]

        # Get the max range for each axis we iterate over
        index_axis_max_range = [max_axis_range[axis_name] for axis_name in index_axis_names]

        # Get the dataframe where each row is a tuple of indices
        index_df = make_df_from_ranges(index_axis_max_range, index_axis_names)

        flattened_output = False
    else:
        # A dataframe of indices was provided. Verify that we did not *also* receive index_axis_names
        assert index_axis_names is None
        index_axis_max_range = index_df.max().to_list()

        flattened_output = True

    # Create an empty tensor to show the patched metric for each patch
    if flattened_output:
        patched_metric_output = torch.zeros(len(index_df), device=model.cfg.device)
    else:
        patched_metric_output = torch.zeros(index_axis_max_range, device=model.cfg.device)

    # A generic patching hook - for each index, it applies the patch_setter appropriately to patch the activation
    def patching_hook(corrupted_activation, hook, index, clean_activation):
        return patch_setter(corrupted_activation, index, clean_activation)

    # Iterate over every list of indices, and make the appropriate patch!
    for c, index_row in enumerate(tqdm((list(index_df.iterrows())))):
        index = index_row[1].to_list()

        # The current activation name is just the activation name plus the layer (assumed to be the first element of the input)
        current_activation_name = utils.get_act_name(activation_name, layer=index[0])

        # The hook function cannot receive additional inputs, so we use partial to include the specific index and the corresponding clean activation
        current_hook = partial(
            patching_hook,
            index=index,
            clean_activation=clean_cache[current_activation_name],
        )

        # Run the model with the patching hook and get the logits!
        patched_logits = model.run_with_hooks(
            corrupted_tokens, fwd_hooks=[(current_activation_name, current_hook)]
        )

        # Calculate the patching metric and store
        if flattened_output:
            patched_metric_output[c] = patching_metric(patched_logits).item()
        else:
            patched_metric_output[tuple(index)] = patching_metric(patched_logits).item()

    if return_index_df:
        return patched_metric_output, index_df
    else:
        return patched_metric_output


# %%
# Defining patch setters for various shapes of activations
def layer_pos_patch_setter(corrupted_activation, index, clean_activation):
    """
    Applies the activation patch where index = [layer, pos]

    Implicitly assumes that the activation axis order is [batch, pos, ...], which is true of everything that is not an attention pattern shaped tensor.
    """
    assert len(index) == 2
    layer, pos = index
    corrupted_activation[:, pos, ...] = clean_activation[:, pos, ...]
    return corrupted_activation