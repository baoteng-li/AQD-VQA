import torch, string
import argparse
import requests
from PIL import Image
from io import BytesIO
import io
import json, os
import pandas as pd
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

    images_data = pd.read_parquet(args.images_path)
    images_data_id = images_data["id"]
    bytes_data = images_data["image"].apply(lambda x: x["bytes"])

    t = 0
    f = 0
    results = []
    for id in tqdm(json_data):
        item = json_data.get(id)
        image_id = item.get('imageId')
        question = item.get('question')
        answer = item.get('answer')

        for id in range(len(images_data_id)):
            image_data_id = images_data_id[id]
            if image_data_id == image_id:
                image_path = os.path.join(args.images_folder, f"{image_id}.jpg")

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
    parser.add_argument("--images_path", type=str, default='GQA/testdev_balanced_images/testdev-00000-of-00001.parquet')
    parser.add_argument("--images_folder", type=str, default='gqa_testdev_images')
    parser.add_argument("--json_path", type=str, default='GQA/question1.2/testdev_balanced_questions.json')
    parser.add_argument("--save_file", type=str, default="result/gqa_testdev_qwen.json")
    parser.add_argument("--model_path", type=str, default="Qwen-VL-Chat")
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()

    eval(args)
