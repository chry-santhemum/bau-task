import random
import json
from concurrent.futures import ProcessPoolExecutor
import math
from functools import partial

TYPES_SIMPLE = {
    "fruit": ["apple", "banana", "strawberry", "pear", "grape", "watermelon", "pineapple", "mango", "blueberry", "peach"],
    "animal": ["dog", "cat", "lion", "elephant", "giraffe", "monkey", "penguin", "dolphin", "tiger", "bear"],
    "tree": ["oak", "ginkgo", "pine", "birch", "willow", "redwood", "fir", "spruce", "palm", "sequoia"],
    "sport": ["soccer", "basketball", "tennis", "baseball", "swimming", "volleyball", "golf", "skiing", "cricket", "hockey"],
    "country": ["USA", "Canada", "Mexico", "Brazil", "UK", "France", "Germany", "China", "India", "Australia"],
}

TYPES = {
    "fruit": ["apple", "banana", "strawberry", "pear", "grape", "watermelon", "pineapple", "mango", "blueberry", "peach"],
    "animal": ["dog", "cat", "lion", "elephant", "giraffe", "monkey", "penguin", "dolphin", "tiger", "bear"],
    "tree": ["oak", "maple", "pine", "birch", "willow", "redwood", "fir", "spruce", "palm", "sequoia"],
    "weather": ["rain", "snow", "sunshine", "wind", "fog", "hail", "sleet", "thunderstorm", "blizzard", "drizzle"],
    "color": ["red", "blue", "green", "yellow", "purple", "black", "white", "pink", "brown", "cyan"],
    "emotion": ["happiness", "sadness", "anger", "fear", "surprise", "disgust", "joy", "excitement", "anxiety", "contentment"],
    "sport": ["soccer", "basketball", "tennis", "baseball", "swimming", "volleyball", "golf", "skiing", "cricket", "hockey"],
    "human body part": ["heart", "brain", "lung", "liver", "kidney", "stomach", "hand", "foot", "eye", "ear"],
    "geometric shape": ["circle", "square", "triangle", "rectangle", "oval", "hexagon", "cube", "sphere", "pyramid", "cylinder"],
    "country": ["USA", "Canada", "Mexico", "Brazil", "UK", "France", "Germany", "China", "India", "Australia"],
    # "household item": ["chair", "table", "bed", "lamp", "sofa", "television", "microwave", "fridge", "cupboard", "toilet"],
    # "celestial body": ["Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto", "Ganymede", "Ceres"]
    # "gemstone": ["diamond", "ruby", "emerald", "sapphire", "amethyst", "topaz", "opal", "garnet", "turquoise", "pearl"],
    # "building": ["house", "skyscraper", "school", "hospital", "library", "museum", "stadium", "castle", "church", "airport"],
    # "flower": ["rose", "tulip", "daisy", "sunflower", "lily", "orchid", "dandelion", "violet", "poppy", "marigold"],
    # "dessert": ["cake", "ice cream", "cookie", "pie", "brownie", "pudding", "donut", "cupcake", "cheesecake", "tiramisu"],
    # "vegetable": ["carrot", "broccoli", "spinach", "potato", "eggplant", "kale", "celery", "lettuce", "onion", "garlic"],
    # "music genre": ["rock", "pop", "jazz", "classical", "hip hop", "electronic", "country", "reggae", "blues", "folk"],
    # "book": ["novel", "biography", "autobiography", "textbook", "poetry collection", "graphic novel", "anthology", "memoir", "dictionary", "encyclopedia"],
    # "public transport": ["bus", "train", "subway", "tram", "ferry", "trolleybus", "cable car", "light rail", "monorail", "water taxi"],
    # "gemstone": ["diamond", "ruby", "emerald", "sapphire", "amethyst", "topaz", "opal", "garnet", "turquoise", "pearl"],
    # "building": ["house", "skyscraper", "school", "hospital", "library", "museum", "stadium", "castle", "church", "airport"],
}

random.seed(142857)


def craft_question(num_total_items: int, target_num: int | None = None) -> tuple[str, list[str], list[int], int]:
    """
    Returns information needed to build a random prompt.
    """
    target_type = random.choice(list(TYPES.keys()))

    # select target type
    target_items = TYPES[target_type]
    if target_num is None:
        target_num = random.randint(1, min(len(target_items), num_total_items-1))
    assert target_num > 0 and target_num <= len(target_items)
        
    selected = random.sample(target_items, target_num)
    remaining_slots = num_total_items - target_num

    # select other types
    all_other_items = []
    for other_type, items in TYPES.items():
        if other_type != target_type:
            all_other_items.extend(items)

    assert remaining_slots <= len(all_other_items), f"Not enough items to fill the remaining slots: {remaining_slots} > {len(all_other_items)}"
    other = random.sample(all_other_items, remaining_slots)
    
    # combine and remember the indices of selected items
    item_list = selected + other
    selected_indices = []

    random.shuffle(item_list)
    for idx, item in enumerate(item_list):
        if item in selected:
            selected_indices.append(idx)
    
    return target_type, item_list, selected_indices, target_num


def make_prompt(target_type: str, item_list: list[str], instruct=True) -> str:
    if instruct:
        return f"Count the number of words in the following list that match the given type.\nType: {target_type}\nList: {' '.join(item_list)}\nOnly output the final answer."
    else:
        return f"Count the number of words in the following list that match the given type, and put the numerical answer in parentheses.\nType: {target_type}\nList: {' '.join(item_list)}\nAnswer: ("


def get_dataset_example(num_total_items: int, instruct=True) -> dict:
    target_type, item_list, selected_indices, target_num = craft_question(num_total_items)

    ex = {
        "target_type": target_type,
        "items": item_list,
        "indices": selected_indices,
        "answer": target_num,
        "text": make_prompt(target_type, item_list, instruct)
    }
    return ex


def get_dataset_one_worker(num_examples: int, mean_total_items: int = 7, std_total_items: int = 2, instruct=True):
    ds = []
    for _ in range(num_examples):
        # choose total number with Gaussian distribution
        num_total_items = round(random.normalvariate(mean_total_items, std_total_items))
        num_total_items = max(2, min(15, num_total_items))
        example = get_dataset_example(num_total_items, instruct=instruct)
        ds.append(example)
    return ds


def save_dataset(num_examples: int, save_path: str, difficulty: str="medium", num_workers: int = 8, instruct=True):
    """
    save dataset to a jsonl file, and print out a few examples
    """
    if difficulty == "easy":
        mean_total_items = 5
        std_total_items = 1
    elif difficulty == "medium":
        mean_total_items = 7
        std_total_items = 2
    elif difficulty == "hard":
        mean_total_items = 9
        std_total_items = 2
    else:
        raise ValueError(f"Invalid difficulty: {difficulty}")
    
    # split tasks
    chunk_size = math.ceil(num_examples / num_workers)
    chunks = [min(chunk_size, num_examples - i * chunk_size) for i in range(num_workers)]
    
    # generate in parallel
    ds = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(partial(get_dataset_one_worker, mean_total_items=mean_total_items, std_total_items=std_total_items, instruct=instruct), chunks))
        for chunk in results:
            ds.extend(chunk)
    
    with open(save_path, 'w') as f:
        for example in ds:
            f.write(json.dumps(example) + '\n')
    
    # print a few examples
    print(f"\nGenerated {len(ds)} examples and saved to {save_path}")
    print("\nFirst 5 examples:")
    for example in ds[:5]:
        print(example['text'])


if __name__ == "__main__":
    save_dataset(num_examples=5000, save_path="dataset/dataset_hard.jsonl", difficulty="hard", num_workers=16, instruct=False)
