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

SYSTEM_MESSAGE = """You are a helpful AI assistant. Your task is to transform the input tuples into sub-questions based on the original question.
tuples formats are: 
1 | Entity - type(entity) - question word
2 | Attribute - type(entity, attribute) - question word
3 | Relation - type(entity, relation, entity) - question word
The format of the output sub-questions are:
1 | sub-questions 1
2 | sub-questions 2
3 | sub-questions 3
Note: 
1. According to the type conversion sub-questions of the question word cued at the end of the tuple, the sub-questions should be related to the original question.
2. A one-to-one correspondence between sub-questions and tuples is guaranteed.
3. Sub-questions should not duplicate with each other.
4. Output strictly according to the formats and examples, and do not output redundant content!
Please refer to the following example:
"""

FEW_SHOT_EXAMPLES = [
    {
        "role": "user", "content": "Why are there so many cats in this store?\n1 | Entity - whole(cats) - what\n2 | Entity - whole(store) - what\n3 | Attribute - number(cats, so many) - why\n4 | Relation - location(cats, in, store) - why"
    },
    {
        "role": "assistant", "content": "1 | What kind of cats are these?\n2 | What kind of store is this?\n3 | Why are there so many cats?\n4 | Why are these cats in the store?"
    },
    {
        "role": "user", "content": "Has the Mona Lisa ever been stolen?\n1 | Entity - whole(Mona Lisa) - what\n2 | Attribute - state(Mona Lisa, been stolen) - whether"
    },
    {
        "role": "assistant", "content": "1 | What is Mona Lisa?\n2 | Was the Mona Lisa stolen?"
    },
    {
        "role": "user", "content": "What do these people do in the store?\n1 | Entity - whole(people) - what\n2 | Entity - whole(store) - what\n3 | Relation - action(people, what, store) - what"
    },
    {
        "role": "assistant", "content": "1 | What do these people do?\n2 | What kind of store is this?\n3 | What do people do in stores?"
    },
    {
        "role": "user", "content": "Why is the person in the picture wearing orange clothes?\n1 | Entity - whole(person) - what\n2 | Entity - whole(clothes) - what\n3 | Attribute - colour(clothes, orange) - why\n4 | Relation - action(person, wearing, clothes) - why"
    },
    {
        "role": "assistant", "content": "1 | What does this person do?\n2 | What kind of dress is this?\n3 | Why is this dress orange?\n4 | Why is this man wearing this dress?"
    },
    {
        "role": "user", "content": "What colour is the car in the distance?\n1 | Entity - whole(car) - what\n2 | Attribute - colour(car, what) - what\n3 | Attribute - location(car, distance) - what"
    },
    {
        "role": "assistant", "content": "1 | What's that car in the distance?\n2 | What color is the car in the distance?\n3 | What is the location of the distant car in the picture?"
    },
    {
        "role": "user", "content": "Are these people competing in world-class competitions?\n1 | Entity - whole(people) - what\n2 | Entity - whole(competitions) - what\n3 | Attribute - level(competitions, world-class) - whether\n4 | Relation - action(people, competing, competitions) - whether"
    },
    {
        "role": "assistant", "content": "1 | What do these people do?\n2 | What competition is this?\n3 | Is it world-class competitions?\n4 | Whether these people competing in competitions?"
    },
    {
        "role": "user", "content": "What's the final score of the basketball game?\n1 | Entity - whole(basketball game) - what\n2 | Attribute - result(basketball game, final score) - what"
    },
    {
        "role": "assistant", "content": "1 | What basketball game is this?\n2 | What's the final score in a basketball game?"
    },
    {
        "role": "user", "content": "What brand of watch is the man wearing on his wrist?\n1 | Entity - whole(man) - what\n2 | Entity - whole(watch) - what\n3 | Entity - part(man, wrist) - what\n4 | Attribute - brand(watch, what) - what\n5 | Relation - action(man, wearing, watch) - what"
    },
    {
        "role": "assistant", "content": "1 | What does this man do?\n2 | What kind of watch is this?\n3 | What are the wrists of this man?\n4 | What brand is the watch?\n5 | What watch does the man wear?"
    },
]

def llama3_completion(
    prompt,
    return_response=False,
    max_new_tokens=512,
):
    # messages = [
    #     {"role": "user", "content": prompt}
    # ]

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
    data_file = 'result/tuple_decom_aokvqa_val.json'
    save_file = 'result/sub_questions_aokvqa_val.json'
    llama3_path = "Meta-Llama-3-8B-Instruct"
    device = 'cuda'

    pipeline = transformers.pipeline(
        "text-generation",
        model=llama3_path,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=device
    )

    with open(data_file, 'r') as file:
        datas = json.load(file)
    
    results = []
    for id in tqdm(range(len(datas))):
        image_path=datas[id].get("image_path")
        question=datas[id].get("question")
        tuples=datas[id].get("tuples")

        description = question
        for j in range(len(tuples)):
            description=description+"\n"+tuples[j]

        output = llama3_completion(description)

        sub_questions = output.strip().split('\n')
        result = dict(
            image_path=image_path,
            question=question,
            tuples=tuples,
            sub_questions=sub_questions
        )

        results.append(result)

        if len(results) == 10:
            with open(save_file,"w") as f:
                json.dump(results, f, indent=1)

    with open(save_file,"w") as f:
        json.dump(results, f, indent=1)