## Contents

- `data.py`: script to generate dataset, has 3 difficulties (based on the average number of items in the list in the prompts)

- `benchmarking.py` and `eval_script.sh`: evaluating some open LLMs on the dataset. Results are in `eval/{difficulty}/results.json`. I only used instruct fine-tuned models since they were better at the task.

- `patching.py`: code for patching experiments. Some patching graphs are in `patching_vis/`.

## Patching results

First to make things clean we make sure that all the words involved are tokenized as just one token (with a space in front). Also, Gemma 2 9B was decently good at this task but definitely not robust (77.6% on the easy set), so the results are a bit noisy.

There are two types of algorithms that the model might implement:

- Parallel algorithm: At some layer, all the token positions which belong to the given class attend to the ones before it, and write the running count in its residual stream. The model then uses the running count at the last position to output the answer.

- Sequential algorithm: At layer L1, the second token which belongs to the class attends to the first token, and writes a count of 2 in its residual stream. At some subsequent layer L2, the third token attends to the second token and increments the count of 3, etc. The model then uses the count at the last position to output the answer.

I wasn't able to obtain complete proof that the model is doing one of them and not doing the other, but my guess is that Gemma 2 9B (the model I'm looking at) does the sequential algorithm.

Suppose I have a list of items, whose status of whether or not they belong to the given class is `list_1 = [False, True, False, True, False, True]`. We patch to the model's computation on this list, from the activations on another list `list_2 = [True, False, True, False, True, False]`. At the token indices 1, 3, and 5, if the information stored in the residual stream is "this token doesn't belong to the class", then the patched model would think that there is one fewer item belonging to that class, and so its logits of outputing 3 would get raised. On the other hand if the information in only keeping track of a running sum, then patching wouldn't matter at all for the model's final conclusion.

The results of 10 of these patching runs are in `patching_vis/results`. In general it seems that the layers at which information is moved away from token indices 1, 3, 5, and 7 are sequential (roughly layers 15, 20, 24, 26), not in the same layer. This fits the patterns of the sequential algorithm.

However this isn't conclusive evidence against the parallel algorithm, since it is possible that at each token the model has the running count stored at some layer, but also superimposed with the information for whether the token itself is of the class. And it's difficult to tell if the model is doing both algorithms in superposition without looking at attention patterns, which I would try if I had more time.

