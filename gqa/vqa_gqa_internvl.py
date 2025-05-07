import torch, string
import argparse
import requests
from PIL import Image
from io import BytesIO
import json, os
from tqdm import tqdm
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

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
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=256, do_sample=True)

    with open(args.json_path, 'r') as file:
        json_data = json.load(file)
    len_data = len(json_data)
    print(len_data)

    images_data = pd.read_parquet(args.images_path)
    images_data_id = images_data["id"]

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
        
        pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()

        prompt = question

        output = model.chat(tokenizer, pixel_values, prompt, generation_config)

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
    parser.add_argument("--save_file", type=str, default="result/gqa_testdev_internvl2.json")
    parser.add_argument("--model_path", type=str, default='InternVL2-8B')
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()

    eval(args)
