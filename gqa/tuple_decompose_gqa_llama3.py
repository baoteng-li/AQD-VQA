import re
import json, os
import pandas as pd
import numpy as np
import torch, string
import transformers
from collections import Counter, defaultdict
from pprint import pprint
from typing import List
from nltk import word_tokenize
from tqdm import tqdm

SYSTEM_MESSAGE = """
You are a helpful AI assistant. Your task is to break down the questions into tuples, which are divided into Entity, Attribute, and Relation, using question words to fill in the missing information in the questions.
Their formats are: 
1 | Entity - type(entity) - question word
2 | Attribute - type(entity, attribute) - question word
3 | Relation - type(entity, relation, entity) - question word
Note: 
1. The input question is asked for some images. For questions such as 'What is the breed of cat in the image?', do not decompose the 'image' into tuples.
2. Question words are used at the end of the tuple to suggest the question type. Like 'what, who, why, how, where'. For entity categories, "what" is used to suggest, and Attribute and Relation categories use question words that are consistent with the original question.
3. The decomposed tuple should contain everything in the question.
4. Output strictly according to the formats and examples, and do not output redundant content!
Please refer to the following example:
"""

FEW_SHOT_EXAMPLES = [
    {
        "role": "user", "content": "How many dogs are there in the picture?"
    },
    {
        "role": "assistant", "content": "1 | Entity - whole(dogs) - what\n2 | Attribute - number(dogs, how many) - how many"
    },
    {
        "role": "user", "content": "Why are there so many cats in this store?"
    },
    {
        "role": "assistant", "content": "1 | Entity - whole(cats) - what\n2 | Entity - whole(store) - what\n3 | Attribute - number(cats, so many) - why\n4 | Relation - location(cats, in, store) - why"
    },
    {
        "role": "user", "content": "Has the Mona Lisa ever been stolen?"
    },
    {
        "role": "assistant", "content": "1 | Entity - whole(Mona Lisa) - what\n2 | Attribute - state(Mona Lisa, been stolen) - whether"
    },
    {
        "role": "user", "content": "Why is the person in the picture wearing orange clothes?"
    },
    {
        "role": "assistant", "content": "1 | Entity - whole(person) - what\n2 | Entity - whole(clothes) - what\n3 | Attribute - colour(clothes, orange) - why\n4 | Relation - action(person, wearing, clothes) - why"
    },
    {
        "role": "user", "content": "What colour is the car in the distance?"
    },
    {
        "role": "assistant", "content": "1 | Entity - whole(car) - what\n2 | Attribute - colour(car, what) - what\n3 | Attribute - location(car, distance) - what"
    },
    {
        "role": "user", "content": "Are these people competing in world-class competitions?"
    },
    {
        "role": "assistant", "content": "1 | Entity - whole(people) - what\n2 | Entity - whole(competitions) - what\n3 | Attribute - level(competitions, world-class) - whether\n4 | Relation - action(people, competing, competitions) - whether"
    },
    {
        "role": "user", "content": "What's the final score of the basketball game?"
    },
    {
        "role": "assistant", "content": "1 | Entity - whole(basketball game) - what\n2 | Attribute - result(basketball game, final score) - what"
    },
    {
        "role": "user", "content": "What brand of watch is the man wearing on his wrist?"
    },
    {
        "role": "assistant", "content": "1 | Entity - whole(man) - what\n2 | Entity - whole(watch) - what\n3 | Entity - part(man, wrist) - what\n4 | Attribute - brand(watch, what) - what\n5 | Relation - action(man, wearing, watch) - what"
    },
]

def llama3_completion(
    prompt,
    return_response=False,
    max_new_tokens=512,
):
    messages = []
    messages.append({"role": "system", "content": SYSTEM_MESSAGE})
    messages.extend(FEW_SHOT_EXAMPLES)
    messages.append({"role": "user", "content": prompt})

    prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    if return_response:
        return outputs

    return outputs[0]["generated_text"][len(prompt):]

if __name__ == "__main__":
    data_file = 'GQA/question1.2/testdev_balanced_questions.json'
    save_file = 'result/tuple_gqa_test.json'
    llama3_path = 'Meta-Llama-3-8B-Instruct'
    device = 'cuda'

    pipeline = transformers.pipeline(
        "text-generation",
        model=llama3_path,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=device
    )

    with open(data_file, 'r') as file:
        datas = json.load(file)
    print(len(datas))
    
    results = []
    for id in tqdm(datas):
        item = datas.get(id)
        image_id = item.get('imageId')
        question = item.get('question')
        answer = item.get('answer')

        output = llama3_completion(question)
        tuples = output.strip().split('\n')

        tuples_filtered = []
        for item in tuples:
            if len(item.split("-")) == 3:
                tuples_filtered.append(item)

        result = dict(
                image_id=image_id,
                question=question,
                tuples=tuples_filtered
            )

        results.append(result)

        if len(results) == 10:
            with open(save_file,"w") as f:
                json.dump(results, f, indent=1)

    with open(save_file,"w") as f:
        json.dump(results, f, indent=1)