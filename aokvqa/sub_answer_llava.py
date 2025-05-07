from llava.model.builder import load_pretrained_model
from llava.eval.run_llava import eval_model
import torch, string
import argparse
import requests
from PIL import Image
from io import BytesIO
import json, os
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
import time
from tqdm import tqdm

_QUESTION_TEMPLATE = string.Template("""
These are the additional information provided:
$sub_qa
Based on the picture and additional information, answer this question:
$input_questions
""".strip())

_QUESTION_TEMPLATE_FIRST = string.Template("""
Answer questions based on images:
$input_questions
""".strip())

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def eval(args):
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=model_name,
        device=args.device
    )

    with open(args.json_path, 'r') as file:
        json_data = json.load(file)
    len_data = len(json_data)
    print(len_data)

    results = []
    for i in tqdm(range(len(json_data))):
        result = json_data[i]
        image_path = json_data[i].get("image_path")
        sub_questions_o = json_data[i].get("sub_questions")
        tuples_o = json_data[i].get("tuples")
        
        image = load_image(image_path)

        tuples = []
        sub_questions = []
        if len(tuples_o) == len(sub_questions_o):
            tuples = tuples_o
            sub_questions = sub_questions_o
        else:
            for tuple in tuples_o:
                tuple_id = tuple.split(" | ")[0]
                for sub_question in sub_questions_o:
                    sub_question_id = sub_question.split(" | ")[0]
                    if tuple_id == sub_question_id:
                        tuples.append(tuple)
                        sub_questions.append(sub_question)

        if args.reverse == True:
            sub_questions.reverse()
            tuples.reverse()

        context = []
        index = []
        for id in range(len(tuples)):
            try:
                context.append(tuples[id].split("(")[1].split(")")[0])
                index.append(id)
            except Exception as e:
                print(e)
                context.append(None)
                index.append(id)

        sub_answer = []
        for j in range(len(sub_questions)):
            if " | " in sub_questions[j]:
                question = sub_questions[j].split(" | ")[1]
            else:
                question = sub_questions[j]

            rely = []
            if j in index:
                for id in range(j):
                    if context[id] == None:
                        continue
                    elements = context[id].strip().split(", ")
                    for element in elements:
                        try:
                            if (element in context[j]) & (index[id] not in rely):
                                rely.append(index[id])
                        except Exception as e:
                            print(e)

            sub_qa = ""
            if len(rely) == 0:
                prompt = _QUESTION_TEMPLATE_FIRST.substitute(input_questions=question)
            else:
                for rely_idx in rely:
                    if " | " in sub_questions[rely_idx]:
                        sub_qa = sub_qa+sub_questions[rely_idx].split(" | ")[1]+"\n"
                    else:
                        sub_qa = sub_qa+sub_questions[rely_idx]+"\n"
                prompt = _QUESTION_TEMPLATE.substitute(sub_qa=sub_qa, input_questions=question)

            args_llava = type('Args', (), {
                "model_path": args.model_path,
                "model_base": None,
                "model_name": model_name,
                "query": prompt,
                "conv_mode": None,
                "image_file": None,
                "sep": ",",
                "temperature": 0,
                "top_p": None,
                "num_beams": 1,
                "max_new_tokens": 512
            })()
            output = eval_model(args_llava, tokenizer, model, image_processor, model_name, image)

            sub_answer.append(output)

        if args.reverse == True:
            sub_questions.reverse()
            tuples.reverse()
            sub_answer.reverse()

        result["sub_answer"] = sub_answer
        result["sub_questions"] = sub_questions
        result["tuples"] = tuples
        results.append(result)

        if len(results) == 10:
            with open(args.save_file,"w") as f:
                json.dump(results, f, indent=1)

    print(args.save_file)
    with open(args.save_file,"w") as f:
        json.dump(results, f, indent=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default='result/sub_questions_aokvqa_val.json')
    parser.add_argument("--save_file", type=str, default='result/sub_answer_aokvqa_val_llava_13b.json')
    parser.add_argument("--model_path", type=str, default='llava-v1.5-13b')
    parser.add_argument("--reverse", type=bool, default=True)
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()

    eval(args)
