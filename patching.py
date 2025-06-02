# %%
import torch
import random
from transformer_lens import patching, HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer
import pandas as pd
import plotly.express as px
from data import make_prompt
from benchmarking import add_generation_prompt

from functools import partial
from transformer_lens.patching import generic_activation_patch, layer_pos_patch_setter

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(142857)

TYPES_SIMPLE = {
    "fruit": ["apple", "banana", "strawberry", "pear", "grape", "watermelon", "pineapple", "mango", "blueberry", "peach"],
    "animal": ["dog", "cat", "lion", "elephant", "giraffe", "monkey", "penguin", "dolphin", "tiger", "bear"],
    "sport": ["soccer", "basketball", "tennis", "baseball", "swimming", "volleyball", "golf", "skiing", "cricket", "hockey"],
    "country": ["USA", "Canada", "Mexico", "Brazil", "UK", "France", "Germany", "China", "India", "Australia"],
}

# %%
# model loading takes a while
model_name = "google/gemma-2-9b-it"
model = HookedTransformer.from_pretrained_no_processing(
    model_name,
    torch_dtype=torch.bfloat16,
    device=device,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%
# check tokenization of all words
for type_name, words in TYPES_SIMPLE.items():
    type_tok = model.to_tokens(type_name, prepend_bos=False)
    if len(type_tok) > 1:
        print(f"{type_name}: {type_tok}")

    for word in words:
        word_tok = model.to_tokens(word, prepend_bos=False)
        if len(word_tok) > 1:
            print(f"{type_name}: {word} -> {word_tok}")


# %%
def make_logit_diff_metric(source_ans: int, target_ans: int):
    source_ans_tok_id = model.to_tokens(str(source_ans), prepend_bos=False)[0]
    target_ans_tok_id = model.to_tokens(str(target_ans), prepend_bos=False)[0]

    def logit_diff_metric_(logits):
        logits = logits.detach().float()
        source_ans_prob_B = logits[:,-1].softmax(dim=-1)[:, source_ans_tok_id]
        target_ans_prob_B = logits[:,-1].softmax(dim=-1)[:, target_ans_tok_id]
        # print(f"source prob: {source_ans_prob_B.item()}, target prob: {target_ans_prob_B.item()}")
        return (target_ans_prob_B - source_ans_prob_B).mean()
    
    return logit_diff_metric_

def make_target_ans_metric(target_ans: int):
    target_ans_tok_id = model.to_tokens(str(target_ans), prepend_bos=False)[0]

    def target_ans_metric_(logits):
        target_ans_prob_B = logits[:,-1].softmax(dim=-1)[:, target_ans_tok_id]
        return target_ans_prob_B.mean()
    
    return target_ans_metric_


def test_metric():
    metric = make_logit_diff_metric(7, 4)
    
    prompt = "13 - 9 = "
    with torch.no_grad():
        logits = model(prompt, return_type="logits")
        print(logits.shape)

    print(metric(logits).item())

test_metric()

# %%

def custom_patching(
    model: HookedTransformer, 
    target_tokens,  # can be batched
    source_cache,
    metric,
    act_name:str,
    start_pos=0, 
    end_pos=None, 
    start_layer=0, 
    end_layer=None
):
    if end_layer is None:
        end_layer = model.cfg.n_layers - 1
    if end_pos is None:
        end_pos = len(target_tokens[0]) - 1

    # Create all combinations of layer and pos
    layers = []
    positions = []
    for layer in range(start_layer, end_layer + 1):
        for pos in range(start_pos, end_pos + 1):
            layers.append(layer)
            positions.append(pos)

    df = pd.DataFrame({
        'layer': layers,
        'pos': positions
    })
    
    patching_fn = partial(
        generic_activation_patch,
        patch_setter=layer_pos_patch_setter,
        activation_name=act_name,
        index_df=df,
    )

    results = patching_fn(model, target_tokens, source_cache, metric)
    return results



def items_list_patching(
    source_lists: list[list[str]] | list[str],  # batched
    target_lists: list[list[str]] | list[str],  # batched
    type_names: list[str] | str,  # batched
    model: HookedTransformer,
    tokenizer: AutoTokenizer,
    act_name: str,
    metric,
    **kwargs,
):
    if isinstance(source_lists, list):
        source_lists = [source_lists]
    if isinstance(target_lists, list):
        target_lists = [target_lists]
    if isinstance(type_names, str):
        type_names = [type_names]

    source_prompts = [add_generation_prompt(make_prompt(type_name, source_list, instruct=True), tokenizer, model_thinks=False) for source_list, type_name in zip(source_lists, type_names)]
    target_prompts = [add_generation_prompt(make_prompt(type_name, target_list, instruct=True), tokenizer, model_thinks=False) for target_list, type_name in zip(target_lists, type_names)]

    source_tokens = model.to_tokens(source_prompts, prepend_bos=False)
    target_tokens = model.to_tokens(target_prompts, prepend_bos=False)

    # start patching from start_pos
    start_pos_tok = tokenizer.encode("List", add_special_tokens=False)[0]
    end_pos_tok = tokenizer.encode("Only", add_special_tokens=False)[0]
    start_pos = target_tokens[0].cpu().tolist().index(start_pos_tok)
    end_pos = target_tokens[0].cpu().tolist().index(end_pos_tok)
    print(f"start_pos: {start_pos}, end_pos: {end_pos}")

    labels = [f"{i}_{tok}" for i, tok in enumerate(model.to_str_tokens(target_tokens[0])) if (i >= start_pos+2 and i <= end_pos)]

    _, source_cache = model.run_with_cache(source_tokens)

    patching_results = custom_patching(
        model,
        target_tokens,
        source_cache,
        metric,
        act_name=act_name,
        start_pos=start_pos+2,
        end_pos=end_pos,
        **kwargs,
    )

    return patching_results, labels

# %%

def make_item_list_pair(
    source_running: list[int],
    target_running: list[int],
    type_name: str | None = None
):
    if type_name is None:
        type_name = random.choice(list(TYPES_SIMPLE.keys()))
    
    type_items = TYPES_SIMPLE[type_name]
    random.shuffle(type_items)
    other_items = []
    for other_type, items in TYPES_SIMPLE.items():
        if other_type != type_name:
            other_items.extend(items)
    random.shuffle(other_items)

    length = len(source_running)
    assert length == len(target_running)

    # build item lists
    source_count, target_count = 0, 0
    source_list, target_list = [], []
    for pos in range(length):
        if source_running[pos] == source_count:
            source_list.append(other_items[pos - source_count])
        elif source_running[pos] == source_count + 1:
            source_count += 1
            source_list.append(type_items[source_count])
        else:
            raise ValueError("List of running scores for source is invalid")
            
        if target_running[pos] == target_count:
            target_list.append(other_items[pos - target_count])
        elif target_running[pos] == target_count + 1:
            target_count += 1
            target_list.append(type_items[target_count])
        else:
            raise ValueError("List of running scores for target is invalid")
        
    return source_list, target_list, type_name

# %%

source_running = [1, 1, 2, 2, 3, 3, 4, 4]
target_running = [0, 1, 1, 2, 2, 3, 3, 4]

source_lists = []
target_lists = []
type_names = []

for i in range(10):
    source_list, target_list, type_name = make_item_list_pair(source_running, target_running, type_name="country")
    print("Source:", source_list)
    print("Target:", target_list)
    print("Type:", type_name)
    # source_lists.append(source_list)
    # target_lists.append(target_list)
    # type_names.append(type_name)

    original_ans = 4
    patched_expect = 3
    act_name = "resid_pre"
    metric = make_logit_diff_metric(patched_expect, original_ans)
    print(f"metric: P[{original_ans}] - P[{patched_expect}]")

    patching_results, labels = items_list_patching(
        source_list,
        target_list,
        type_name,
        model,
        tokenizer,
        act_name=act_name,
        metric=metric,
        end_layer=30,
    )

    fig = px.imshow(
        patching_results.view(31, -1).cpu().numpy(),
        color_continuous_scale="RdBu", 
        x=labels,
        zmin=-1, zmax=1,
        height=700, width=500,
    ) 
    fig.update_xaxes(tickangle=45)
    fig.show()

    fig.write_image(f"patching_vis/results/{''.join([str(i) for i in source_running])}->{''.join([str(i) for i in target_running])}_{act_name}_{original_ans}_{patched_expect}_run_{i}.png")


# %%
