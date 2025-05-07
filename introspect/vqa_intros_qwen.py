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

def clean_data(answer):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(answer)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    lemmatized_answer = ' '.join(lemmatized_tokens)
    cleaned_answer = lemmatized_answer.lower()
    return cleaned_answer

def gqa_acc_compute(model_answer, ref_answer):
    cleaned_ref_answer = clean_data(ref_answer)
    cleaned_model_answer = clean_data(model_answer)

    if 'not possible' in cleaned_model_answer or 'impossible' in cleaned_model_answer:
        result = False
    else:
        if cleaned_ref_answer in cleaned_model_answer:
            result = True
        else:
            result = False

    return result

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

    t = 0
    f = 0
    results = []
    for id in tqdm(json_data):
        item = json_data.get(id)
        image_id = item.get('image_id')
        question = item.get('reasoning_question')
        answer = item.get('reasoning_answer_most_common')

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

        prompt = question

        query = tokenizer.from_list_format([
            {'image': image_path},
            {'text': prompt},
        ])
        output, history = model.chat(tokenizer, query=query, history=None)

        result = gqa_acc_compute(output, answer)
        if result == True:
            t += 1
        else:
            f += 1

        results.append(dict(
            image_id=image_id,
            file_name=file_name,
            question=question,
            answer=answer,
            ik_question=result,
            llava_answer=output
        ))

    print(t, f, t/len_data)
    with open(args.save_file,"w") as f:
        json.dump(results, f, indent=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_folder_path_val", type=str, default='coco/val2017/')
    parser.add_argument("--images_folder_path_train", type=str, default='coco/train2017/')
    parser.add_argument("--json_path", type=str, default='result/VQAIntrospect_val_3000.json')
    parser.add_argument("--save_file", type=str, default="result/intros_val_3000_qwen.json")
    parser.add_argument("--coco_val_path", type=str, default='coco/annotations/captions_val2017.json')
    parser.add_argument("--coco_train_path", type=str, default='coco/annotations/captions_train2017.json')
    parser.add_argument("--model_path", type=str, default="Qwen-VL-Chat")
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()

    eval(args)
