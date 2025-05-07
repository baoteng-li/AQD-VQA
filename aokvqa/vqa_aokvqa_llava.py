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
$input_questions
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

    with open(args.coco_val_path, 'r') as file:
        coco_val = json.load(file)

    with open(args.coco_train_path, 'r') as file:
        coco_train = json.load(file)

    f = 0
    t = 0
    results = []
    correct_choice_list = ["A", "B", "C", "D"]
    acc = []
    for i in tqdm(range(len(json_data))):
        image_id = json_data[i].get("image_id")
        direct_answers = json_data[i].get("direct_answers")
        question = json_data[i].get("question")
        choices = json_data[i].get("choices")
        correct_choice_idx = json_data[i].get("correct_choice_idx")
        correct_choice = correct_choice_list[correct_choice_idx]
        
        file_name = None
        for item in coco_val.get("images"):
            if image_id == item.get("id"):
                file_name = item.get("file_name")
                images_folder_path = args.images_folder_path_val
                break

        if file_name == None:
            for item in coco_train.get("images"):
                if image_id == item.get("id"):
                    file_name = item.get("file_name")
                    images_folder_path = args.images_folder_path_train
                    break

        image_path = os.path.join(images_folder_path, file_name)
        
        try:
            image = load_image(image_path)
        except Exception as e:
            print(e)
        
        prompt = _QUESTION_TEMPLATE.substitute(input_questions=question, choices_a=choices[0], choices_b=choices[1], choices_c=choices[2], choices_d=choices[3])

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

        result = json_data[i]

        ## Multiple Choice setting
        pred = output[0]
        vqa_acc = (correct_choice == pred)
        if vqa_acc == True:
            t+=1
        else:
            f+=1
        acc.append(float(vqa_acc))
        result["ik_question"] = vqa_acc
                
        results.append(result)

    acc = sum(acc) / len(acc) * 100
    print(acc)

    with open(args.save_file,"w") as f:
        json.dump(results, f, indent=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_folder_path_val", type=str, default='coco/val2017/')
    parser.add_argument("--images_folder_path_train", type=str, default='coco/train2017/')
    parser.add_argument("--json_path", type=str, default='aokvqa_v1p0/aokvqa_v1p0_val.json')
    parser.add_argument("--save_file", type=str, default='result/aokvqa_llava_13b.json')
    parser.add_argument("--coco_val_path", type=str, default='coco/annotations/captions_val2017.json')
    parser.add_argument("--coco_train_path", type=str, default='coco/annotations/captions_train2017.json')
    parser.add_argument("--model_path", type=str, default='llava-v1.5-13b')
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()

    eval(args)
