import random
import json
from concurrent.futures import ProcessPoolExecutor
import math

TYPES = {
    "fruit": ["apple", "banana", "strawberry", "pear", "grape", "watermelon", "pineapple", "mango", "blueberry", "peach"],
    "dessert": ["cake", "ice cream", "cookie", "pie", "brownie", "pudding", "donut", "cupcake", "cheesecake", "tiramisu"],
    "vegetable": ["carrot", "broccoli", "spinach", "potato", "eggplant", "kale", "celery", "lettuce", "onion", "garlic"],
    "animal": ["dog", "cat", "lion", "elephant", "giraffe", "monkey", "penguin", "dolphin", "tiger", "bear"],
    "flower": ["rose", "tulip", "daisy", "sunflower", "lily", "orchid", "dandelion", "violet", "poppy", "marigold"],
    "tree": ["oak", "maple", "pine", "birch", "willow", "redwood", "fir", "spruce", "palm", "sequoia"],
    "weather": ["rain", "snow", "sunshine", "wind", "fog", "hail", "sleet", "thunderstorm", "blizzard", "drizzle"],
    "color": ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "pink", "brown"],
    "gemstone": ["diamond", "ruby", "emerald", "sapphire", "amethyst", "topaz", "opal", "garnet", "turquoise", "pearl"],
    "building": ["house", "skyscraper", "school", "hospital", "library", "museum", "stadium", "castle", "church", "airport"],
    "emotion": ["happiness", "sadness", "anger", "fear", "surprise", "disgust", "joy", "excitement", "anxiety", "contentment"],
    "household item": ["chair", "table", "bed", "lamp", "sofa", "television", "microwave", "refrigerator", "washing machine", "vacuum cleaner"],
    "sport": ["soccer", "basketball", "tennis", "baseball", "swimming", "volleyball", "golf", "skiing", "cricket", "hockey"],
    "music genre": ["rock", "pop", "jazz", "classical", "hip hop", "electronic", "country", "reggae", "blues", "folk"],
    "book": ["novel", "biography", "autobiography", "textbook", "poetry collection", "graphic novel", "anthology", "memoir", "dictionary", "encyclopedia"],
    "public transport": ["bus", "train", "subway", "tram", "ferry", "trolleybus", "cable car", "light rail", "monorail", "water taxi"],
    "country": ["United States", "Canada", "Mexico", "Brazil", "United Kingdom", "France", "Germany", "China", "India", "Australia"],
    "human body part": ["heart", "brain", "lung", "liver", "kidney", "stomach", "hand", "foot", "eye", "ear"],
    "beverage": ["water", "coffee", "tea", "milk", "soda", "juice", "wine", "beer", "lemonade", "smoothie"],
    "geometric shape": ["circle", "square", "triangle", "rectangle", "oval", "hexagon", "cube", "sphere", "pyramid", "cylinder"],
    "celestial body": ["Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto", "Ganymede", "Ceres"]
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
        target_num = random.randint(1, min(len(target_items), num_total_items))
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
    items_list = selected + other
    selected_indices = []

    random.shuffle(items_list)
    for idx, item in enumerate(items_list):
        if item in selected:
            selected_indices.append(idx)
    
    return target_type, items_list, selected_indices, target_num


def get_dataset_example(num_total_items: int) -> dict:
    target_type, item_list, selected_indices, target_num = craft_question(num_total_items)

    ex = {
        "target_type": target_type,
        "items": item_list,
        "indices": selected_indices,
        "answer": target_num,
        "text": f"Count the number of words in the following list that match the given type, and put the numerical answer in parentheses.\nType: {target_type}\nList: {' '.join(item_list)}\nAnswer: ("
    }
    return ex


def get_dataset_one_worker(num_examples: int):
    ds = []
    for _ in range(num_examples):
        # choose total number with Gaussian distribution
        num_total_items = round(random.normalvariate(10, 3))
        num_total_items = max(2, min(20, num_total_items))
        example = get_dataset_example(num_total_items)
        ds.append(example)
    return ds


def save_dataset(num_examples: int, save_path: str, num_workers: int = 8):
    """
    save dataset to a jsonl file, and print out a few examples
    """
    
    # split tasks
    chunk_size = math.ceil(num_examples / num_workers)
    chunks = [min(chunk_size, num_examples - i * chunk_size) for i in range(num_workers)]
    
    # generate in parallel
    ds = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(get_dataset_one_worker, chunks))
        for chunk in results:
            ds.extend(chunk)
    
    with open(save_path, 'w') as f:
        for example in ds:
            f.write(json.dumps(example) + '\n')
    
    # print a few examples
    print(f"\nGenerated {len(ds)} examples and saved to {save_path}")
    print("\nFirst 3 examples:")
    for i, example in enumerate(ds[:3]):
        print(f"{example['text']}{example['answer']})")


if __name__ == "__main__":
    save_dataset(num_examples=10, save_path="counting_ds.jsonl", num_workers=8)
