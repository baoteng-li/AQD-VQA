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
import nltk
from nltk.stem import WordNetLemmatizer

_QUESTION_TEMPLATE = string.Template("""
These are the additional information provided:
$sub_qa

$input_questions
Based on the picture and additional information, choose one of the following four choices that answers the question, note that the output answer must be the complete option:
A. $choices_a
B. $choices_b
C. $choices_c
D. $choices_d
""".strip())

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def clean_data(answer):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(answer)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    lemmatized_answer = ' '.join(lemmatized_tokens)
    cleaned_answer = lemmatized_answer.lower()
    return cleaned_answer

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

    with open(args.sub_qa_path, 'r') as file:
        sub_qa_data = json.load(file)

    f = 0
    t = 0
    results = []
    acc = []
    correct_choice_list = ["A", "B", "C", "D"]
    for i in tqdm(range(len(json_data))):
        result = json_data[i]
        image_id = json_data[i].get("image_id")
        question = json_data[i].get("question")
        direct_answers = json_data[i].get("direct_answers")
        choices = json_data[i].get("choices")
        correct_choice_idx = json_data[i].get("correct_choice_idx")
        correct_choice = correct_choice_list[correct_choice_idx]
        image_path = sub_qa_data[i].get("image_path")
        ik_question = json_data[i].get("ik_question")
        sub_questions = sub_qa_data[i].get("sub_questions")
        sub_answer = sub_qa_data[i].get("sub_answer")

        result["tuples"] = sub_qa_data[i].get("tuples")
        result["sub_questions"] = sub_questions
        result["sub_answer"] = sub_answer
        sub_qa = ""
        for j in range(len(sub_questions)):
            try:
                sub_qa = sub_qa+sub_questions[j]+" "+sub_answer[j]+"\n"
            except Exception as e:
                print("The serial numbers do not match: "+str(image_id))
        
        try:
            image = load_image(image_path)
        except Exception as e:
            print(e)
        
        prompt = _QUESTION_TEMPLATE.substitute(sub_qa=sub_qa, input_questions=question, choices_a=choices[0], choices_b=choices[1], choices_c=choices[2], choices_d=choices[3])

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

        ## Multiple Choice setting
        pred = output[0]
        vqa_acc = (correct_choice == pred)
        if vqa_acc == True:
            t += 1
        else:
            f += 1
        acc.append(float(vqa_acc))
        result["final_answer_correct"] = vqa_acc
        result["final_answer"] = output
        
        results.append(result)

    acc = sum(acc) / len(acc) * 100
    print(t, f)
    print(acc)
    print(args.save_file)
    with open(args.save_file,"w") as f:
        json.dump(results, f, indent=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_folder_path", type=str, default='coco/val2017/')
    parser.add_argument("--json_path", type=str, default='aokvqa_v1p0/aokvqa_v1p0_val.json')
    parser.add_argument("--save_file", type=str, default='result/final_answer_aokvqa_val_llava_13b.json')
    parser.add_argument("--sub_qa_path", type=str, default='result/sub_answer_aokvqa_val_llava_13b.json')
    parser.add_argument("--coco_path", type=str, default='coco/annotations/captions_val2017.json')
    parser.add_argument("--model_path", type=str, default='llava-v1.5-13b')
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()

    eval(args)
