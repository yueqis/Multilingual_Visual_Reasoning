import logging
import argparse
import csv
from openai import OpenAI
from PIL import Image
import base64

def parse_args():
    parser = argparse.ArgumentParser(
        description="GPT4V Pipeline"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="CSV containing the test data of MaRVL/NLVR2",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Where to store final output",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI api key",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Index of the start position of MaRVL",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=10,
        help="Number of MaRVL tests to perform",
    )
    args = parser.parse_args()
    # Sanity checks: check whether test file, output file, and model_addr present
    if args.test_file is None:
        raise ValueError(
            "Need test file."
        )
    if args.output_file is None:
        raise ValueError(
            "Need output file."
        )
    return args

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_gpt(client, statement, left_image, right_image):
    prompt = f"Let's think step by step to reason if the statement is True or False based on the left image and the right image. Statement: '{statement}' \nRespond in English using this format: 'Reasoning: .....\nOutput: True.' or 'Reasoning: .....\nOutput: False.'"
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt,},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{left_image}",},},
                    {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{right_image}",},},
                ],
            }
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content

def get_result(response):
    if not response.startswith('Reasoning: '): return response, response
    try: 
        reasoning = response.split('\n')[0].strip()
        output = response.split('\n')[1].strip()
        if output.startswith('Output: True'): return True, reasoning
        if output.startswith('Output: False'): return False, reasoning
    except Exception as e:
        logging.info(f"exception {e} for response {response}")
    try: 
        reasoning = response.split('Output:')[0].strip()
        output = response.split('Output:')[1].strip()
        if output.startswith('True'): return True, reasoning
        if output.startswith('False'): return False, reasoning
    except Exception as e:
        logging.info(f"exception {e} for response {response}")
    if ("True" in response) or ("true" in response): return True, response
    if ("False" in response) or ("false" in response): return False, response
    return response, response

def main():
    args = parse_args()
    logging.info(args)
    logging.info('marvl test file: ' + args.test_file)
    logging.info('output_file: ' + args.output_file)
    logging.info(f"Next time start testing from {args.start+args.length}")
    client = OpenAI(api_key = args.api_key)
    with open(args.test_file, 'r', newline='') as file:
        reader = csv.DictReader(file)
        marvl_rows = list(reader)
    correct = 0
    err = 0
    output = []
    for idx in range(args.start, min(len(marvl_rows), args.start+args.length)):
        try:
            id = marvl_rows[idx]['id']
            statement = marvl_rows[idx]['caption']
            label = marvl_rows[idx]['label'] == "True"
            left_image = encode_image(marvl_rows[idx]['image_1_path'])
            right_image = encode_image(marvl_rows[idx]['image_2_path'])
            response = generate_gpt(client, statement, left_image, right_image)
            logging.info(f"{id}: response - {response}")
            pred, reasoning = get_result(response)
            if not (pred == True or pred == False): 
                err += 1
                logging.info(f"pred of id {id} is {pred}")
            correct = correct + 1 if pred == label else correct
            output.append({'id': id, 'pred': pred, 'reasoning': reasoning})
            logging.info(f"label: {label}, pred: {pred}\nstatement: {statement}")
        except Exception as e:
            err += 1
            logging.info(f"An error occurred in generating output: {str(e)}")
        logging.info(f"Correct: {correct}, Total: {idx + 1 - args.start}, Accuracy: {correct / (idx + 1 - args.start)}, Failed: {err}\n\n")
    with open(args.output_file, mode='a', newline='') as file:
        fieldnames = ['id', 'pred', 'reasoning']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in output:
            writer.writerow(row)
        logging.info("Output stored in " + args.output_file)

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

if __name__ == "__main__":
    main()