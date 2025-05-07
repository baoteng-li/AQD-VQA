import re
import json, os
import pandas as pd
import numpy as np
import torch, string
import transformers
import time
from collections import Counter, defaultdict
from pprint import pprint
from typing import List
from nltk import word_tokenize
from tqdm import tqdm

SYSTEM_MESSAGE = """
Your task: According to the input of the questions with the image, Each question is rated on four dimensions:
1. Objective: Fact-based observations. / Subjective: Based on subjective speculation, may vary from person to person.
2. Attribute: An attribute or property of an object. / Relation: The interaction between multiple objects.
3. Static: About static things. / Dynamic: About the movement, development, change of things.
4. Single Object: About a single thing. / Overall Scene: About the entire scene or event in the image.
Note: 
1. The closer it is to the previous category, the closer it is to the 1, and the closer it is to the later category, the closer it is to the 5.
2. The scores for each category are output sequentially, separated by Spaces.
3. Make sure to output four scores for each question!
4. Output strictly according to the formats and examples, and do not output redundant content!
Please refer to the following example:
"""

FEW_SHOT_EXAMPLES = [
    {
        "role": "user", "content": "How many dogs are there in the picture?"
    },
    {
        "role": "assistant", "content": "1.1 2.1 3.2 4.3"
    },
    {
        "role": "user", "content": "Why are there so many cats in this store?"
    },
    {
        "role": "assistant", "content": "1.4 2.4 3.1 4.5"
    },
    {
        "role": "user", "content": "Is this plane the first plane in the world?"
    },
    {
        "role": "assistant", "content": "1.1 2.2 3.2 4.1"
    },
    {
        "role": "user", "content": "Do you think this person will be sad?"
    },
    {
        "role": "assistant", "content": "1.5 2.2 3.3 4.2"
    },
    {
        "role": "user", "content": "Why is the person in the picture wearing orange clothes?"
    },
    {
        "role": "assistant", "content": "1.3 2.4 3.1 4.2"
    },
    {
        "role": "user", "content": "Are these people competing in world-class competitions?"
    },
    {
        "role": "assistant", "content": "1.1 2.1 3.5 4.5"
    },
]

def class_file_dump(threshold, file_data, classify_data, save_file_name, visable=False):
    s = 0
    c = 0
    s_file_datas = []
    c_file_datas = []
    
    for i in range(len(file_data)):
        file_item = file_data[i]
        complex_idx = 0
        tags = classify_data[i].get("tags")

        if tags == None:
            c_file_datas.append(file_item)
            c += 1
            continue

        try:
            for id in range(len(tags)):
                complex_idx += int(tags[id])
        except Exception as e:
            print(file_data[i].get("file_name"))
        probability = (complex_idx/4)

        if (probability < threshold):
            s_file_datas.append(file_item)
            s += 1
        else:
            c_file_datas.append(file_item)
            c += 1

    if visable == True:
        print('threshold: ', threshold)
        print("number：", s, c)

    save_file = save_file_name+'_s_'+str(threshold)+'.json'
    with open(save_file,"w") as f:
        json.dump(s_file_datas, f, indent=1)

    save_file = save_file_name+'_c_'+str(threshold)+'.json'
    with open(save_file,"w") as f:
        json.dump(c_file_datas, f, indent=1)

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
    data_file = 'aokvqa_v1p0/aokvqa_v1p0_val.json'
    save_file_name = 'result/classify_aokvqa'
    llama3_path = "Meta-Llama-3-8B-Instruct"
    threshold = 1.4
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

    start_time = time.perf_counter()
    
    results = []
    wrong_str = []
    id = 0
    retries = 0
    with tqdm(total=len(datas)) as pbar:
        while id < len(datas):
            image_id = datas[id].get('image_id')
            question = datas[id].get('question')

            output = llama3_completion(question)
            tags_list = output.split('.')[1:]
            tags = []
            for item in tags_list:
                tags.append(item.split(" ")[0])
            if len(tags) != 4:
                wrong_str.append(output)
                retries += 1
                if retries > 5:
                    tags = None
                    result = dict(
                        image_id=image_id,
                        question=question,
                        tags=tags
                    )

                    results.append(result)
                    
                    id += 1
                    retries = 0
                    pbar.update(1)
            else:
                result = dict(
                    image_id=image_id,
                    question=question,
                    tags=tags
                )

                results.append(result)
                
                id += 1
                retries = 0
                pbar.update(1)

    class_file_dump(threshold, datas, results, save_file_name)

    end_time = time.perf_counter()

    print(wrong_str)
    print(f"tuple decompose run time: {end_time - start_time:.6f} second")