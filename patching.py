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
        word_tok = model.to_token(word, prepend_bos=False)
        if len(word_tok) > 1:
            print(f"{type}: {word} -> {word_tok}")
            break


# %%

def make_prompt_pair(num_total_items: int):
    target_type, item_list, selected_indices, target_num = craft_question(num_total_items, min_target_num=2)

    clean_prompt = f"Count the number of words in the following list that match the given type, and put the numerical answer in parentheses.\nType: {target_type}\nList: {' '.join(item_list)}\nAnswer: ("

    index_to_remove = random.sample(selected_indices, 1)[0]
    item_list[index_to_remove] = "corrupt"
    selected_indices.remove(index_to_remove)
    target_num -= 1
    corrupt_prompt = f"Count the number of words in the following list that match the given type, and put the numerical answer in parentheses.\nType: {target_type}\nList: {' '.join(item_list)}\nAnswer: ("

    return clean_prompt, corrupt_prompt



# %%

def make_counting_metric(clean_ans: int, corrupt_ans: int):
    clean_ans_tok_id = model.to_token(clean_ans, prepend_bos=False)
    print(clean_ans_tok_id)
    corrupt_ans_tok_id = model.to_token(corrupt_ans, prepend_bos=False)
    print(corrupt_ans_tok_id)

    def metric_(logits):
        clean_ans_prob_B = logits[:,-1].softmax(dim=-1)[:, clean_ans_tok_id]
        corrupt_ans_prob_B = logits[:,-1].softmax(dim=-1)[:, corrupt_ans_tok_id]
        print(clean_ans_prob_B, corrupt_ans_prob_B)
        return (clean_ans_prob_B - corrupt_ans_prob_B).mean()
    
    return metric_


def test_metric():
    metric = make_counting_metric()

# %%

    
clean_prompt, corrupt_prompt = make_prompt_pair(10)


patching.get_act_patch_mlp_out(
    model,
    corrupt_tokens,
    clean_cache,
    metric
)
