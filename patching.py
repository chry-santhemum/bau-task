# %%
import torch
import random
from transformer_lens import patching, HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer
import pandas as pd

from data import craft_question, TYPES, make_prompt
from benchmarking import add_generation_prompt

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(142857)

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
# check tokenization of all words in TYPES
for type_name, words in TYPES.items():
    for word in words:
        word_tok = model.to_tokens(word, prepend_bos=False)
        if len(word_tok) > 1:
            print(f"{type_name}: {word} -> {word_tok}")


# %%
def make_counting_metric(source_ans: int, target_ans: int):
    source_ans_tok_id = model.to_tokens(str(source_ans), prepend_bos=False)[0]
    target_ans_tok_id = model.to_tokens(str(target_ans), prepend_bos=False)[0]

    def logit_diff_metric_(logits):
        logits = logits.detach().float()
        source_ans_prob_B = logits[:,-1].softmax(dim=-1)[:, source_ans_tok_id]
        target_ans_prob_B = logits[:,-1].softmax(dim=-1)[:, target_ans_tok_id]
        print(f"source prob: {source_ans_prob_B.item()}, target prob: {target_ans_prob_B.item()}")
        return (target_ans_prob_B - source_ans_prob_B).mean()
    
    def target_ans_metric_(logits):
        target_ans_prob_B = logits[:,-1].softmax(dim=-1)[:, target_ans_tok_id]
        return target_ans_prob_B.mean()
    
    return logit_diff_metric_, target_ans_metric_


def test_metric():
    metric, _ = make_counting_metric(7, 4)
    
    prompt = "13 - 9 = "
    with torch.no_grad():
        logits = model(prompt, return_type="logits")
        print(logits.shape)

    print(metric(logits))

test_metric()

# %%

from functools import partial
from transformer_lens.patching import generic_activation_patch, layer_pos_patch_setter


def custom_mlp_patching(
    model, 
    target_tokens,
    source_cache,
    metric,
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
        activation_name="mlp_out",
        index_df=df,
    )

    results = patching_fn(model, target_tokens, source_cache, metric)
    return results


# %%

target_type = "fruit"
source_item_list = ["cat", "apple", "grape", "triangle", "baseball", "car"]
target_item_list = ["cat", "apple", "baseball", "car", "grape", "banana"]
source_ans, target_ans = 0, 0

for item in source_item_list:
    if item in TYPES[target_type]:
        source_ans += 1

for item in target_item_list:
    if item in TYPES[target_type]:
        target_ans += 1


source_prompt = add_generation_prompt(make_prompt(target_type, source_item_list, instruct=True), tokenizer, model_thinks=False)
target_prompt = add_generation_prompt(make_prompt(target_type, target_item_list, instruct=True), tokenizer, model_thinks=False)

start_pos_tok = tokenizer.encode("List", add_special_tokens=False)[0]

source_tokens = model.to_tokens(source_prompt, prepend_bos=False)
target_tokens = model.to_tokens(target_prompt, prepend_bos=False)
start_pos = target_tokens[0].cpu().tolist().index(start_pos_tok)
print(f"start_pos: {start_pos}")

_, source_cache = model.run_with_cache(source_tokens, remove_batch_dim=False)
metric, _ = make_counting_metric(source_ans, target_ans)

patching_results = custom_mlp_patching(
    model,
    target_tokens,
    source_cache,
    metric,
    start_pos=start_pos,
)

# %%

import plotly.express as px

labels = [f"{i}_{tok}" for i, tok in enumerate(model.to_str_tokens(target_tokens)) if i >= start_pos]

print(source_item_list)
print(target_item_list)
fig = px.imshow(
    patching_results.view(model.cfg.n_layers, -1).cpu().numpy(),
    color_continuous_scale="RdBu", 
    x=labels
) 
fig.update_xaxes(tickangle=45)
fig.show()

# %%

# %%

if __name__ == "__main__":
    model_name = "google/gemma-2-9b"
    model = HookedTransformer.from_pretrained_no_processing(
        model_name,
        torch_dtype=torch.bfloat16,
        device=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

