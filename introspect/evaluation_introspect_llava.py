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
import nltk
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

_QUESTION_TEMPLATE = string.Template("""
These are the additional information provided:
$sub_qa
Based on the picture and additional information, answer this question:
$input_questions
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

    with open(args.sub_qa_path, 'r') as file:
        sub_qa_data = json.load(file)

    f = 0
    t = 0
    st = 0
    ct = 0
    results = []
    for i in tqdm(range(len(json_data))):
        result = json_data[i]
        image_id = json_data[i].get("image_id")
        question = json_data[i].get("question")
        answer = json_data[i].get("answer")

        image_path = sub_qa_data[i].get("image_path")
        ik_question = sub_qa_data[i].get("ik_question")
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

        acc_result = gqa_acc_compute(output, answer)
        if acc_result == True:
            t += 1
            result["final_answer_correct"] = True
            if ik_question == True:
                st += 1
            elif ik_question == False:
                ct += 1
        else:
            f += 1
            result["final_answer_correct"] = False
        result["final_answer"] = output
        results.append(result)

    print(t, f, (t/len_data)*100)
    print(args.save_file)
    with open(args.save_file,"w") as f:
        json.dump(results, f, indent=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default='result/VQAIntrospect_val_3000.json')
    parser.add_argument("--save_file", type=str, default='result/final_answer_introspect_val_3000_llava.json')
    parser.add_argument("--sub_qa_path", type=str, default='result/sub_answer_introspect_val_3000_llava.json')
    parser.add_argument("--model_path", type=str, default='llava-v1.5-13b')
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()

    eval(args)
