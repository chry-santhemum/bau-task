## Contents

- `data.py`: script to generate dataset, has 3 difficulties (based on the number of items in the provided list)

- `benchmarking.py` and `eval_script.sh`: evaluating some open LLMs on the dataset. Results are in `eval/{difficulty}/results.json`.

- `patching.py`: code for patching experiments. Some patching graphs are in `patching_vis/`.

## Patching results

First to make things clean we make sure that all the words involved are tokenized as just one token. Then, we take two prompts 

