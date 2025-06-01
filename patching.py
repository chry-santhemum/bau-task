# %%
import torch
import random
from transformer_lens import patching, HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer

from data import craft_question, TYPES

# %%

device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained_no_processing(
    "gemma-2-9b",
    torch_dtype=torch.bfloat16,
    device=device,
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

random.seed(142857)

# %%
# check tokenization of all words in TYPES

for type, words in TYPES.items():
    for word in words:
        print(model.to_token(word, prepend_bos=False))
        break



# %%

def make_prompt_pair(num_total_items: int):
    target_type, items_list, selected_indices, target_num = craft_question(num_total_items, min_target_num=2)

    prompt_clean = f"Count the number of words in the following list that match the given type, and put the numerical answer in parentheses.\nType: {target_type}\nList: {' '.join(item_list)}\nAnswer: ("

    index_to_remove = random.sample(selected_indices, 1)[0]
    items_list[index_to_remove]
    selected_indices.remove(index_to_remove)
    target_num -= 1

    prompt_corrupt = 



# %%
