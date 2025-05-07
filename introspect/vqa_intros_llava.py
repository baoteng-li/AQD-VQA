from llava.model.builder import load_pretrained_model
from llava.eval.run_llava import eval_model
import torch, string
import argparse
import requests
import time
from PIL import Image
from io import BytesIO
import json, os
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer

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

    start_time = time.perf_counter()

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
        try:
            image = load_image(image_path)
        except Exception as e:
            print(e)

        prompt = 'Based on the picture, answer this question: '+question

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

        result = gqa_acc_compute(output, answer)
        if result == True:
            t += 1
        else:
            print(answer, output)
            f += 1
        results.append(dict(
            image_id=image_id,
            file_name=file_name,
            question=question,
            answer=answer,
            ik_question=result,
            llava_answer=output
        ))

    end_time = time.perf_counter()

    print(t, f, t/len_data)
    with open(args.save_file,"w") as f:
        json.dump(results, f, indent=1)

    print(f"sub_a run time: {end_time - start_time:.6f} second")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_folder_path_val", type=str, default='coco/val2017/')
    parser.add_argument("--images_folder_path_train", type=str, default='coco/train2017/')
    parser.add_argument("--json_path", type=str, default='result/VQAIntrospect_val_3000.json')
    parser.add_argument("--save_file", type=str, default="result/intros_val_3000_llava.json")
    parser.add_argument("--coco_val_path", type=str, default='coco/annotations/captions_val2017.json')
    parser.add_argument("--coco_train_path", type=str, default='coco/annotations/captions_train2017.json')
    parser.add_argument("--model_path", type=str, default='llava-v1.5-13b')
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()

    eval(args)
