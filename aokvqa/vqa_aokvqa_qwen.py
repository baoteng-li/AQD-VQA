import torch, string
import argparse
import requests
from PIL import Image
from io import BytesIO
import json, os
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(1234)

_QUESTION_TEMPLATE = string.Template("""
Answer the following questions:
$input_questions
A. $choices_a
B. $choices_b
C. $choices_c
D. $choices_d
""".strip())

def clean_data(answer):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(answer)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    lemmatized_answer = ' '.join(lemmatized_tokens)
    cleaned_answer = lemmatized_answer.lower()
    return cleaned_answer

def eval(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="cuda", trust_remote_code=True).eval()

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
        
        prompt = _QUESTION_TEMPLATE.substitute(input_questions=question, choices_a=choices[0], choices_b=choices[1], choices_c=choices[2], choices_d=choices[3])

        query = tokenizer.from_list_format([
            {'image': image_path},
            {'text': prompt},
        ])
        output, history = model.chat(tokenizer, query=query, history=None)

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
        result["model_answer"] = output
                
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
    parser.add_argument("--save_file", type=str, default='result/aokvqa_qwen_val.json')
    parser.add_argument("--coco_val_path", type=str, default='coco/annotations/captions_val2017.json')
    parser.add_argument("--coco_train_path", type=str, default='coco/annotations/captions_train2017.json')
    parser.add_argument("--model_path", type=str, default='Qwen-VL-Chat')
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()

    eval(args)
