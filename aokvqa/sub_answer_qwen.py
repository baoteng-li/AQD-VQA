import torch, string
import argparse
import requests
from PIL import Image
from io import BytesIO
import json, os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(1234)

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

def eval(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="cuda", trust_remote_code=True).eval()

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
                for id in range(len(sub_questions)):
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

            query = tokenizer.from_list_format([
                {'image': image_path},
                {'text': prompt},
            ])
            output, history = model.chat(tokenizer, query=query, history=None)

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
    parser.add_argument("--save_file", type=str, default='result/sub_answer_aokvqa_val_qwen.json')
    parser.add_argument("--model_path", type=str, default='Qwen-VL-Chat')
    parser.add_argument("--reverse", type=bool, default=True)
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()

    eval(args)
