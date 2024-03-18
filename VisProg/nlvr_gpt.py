import csv
from PIL import Image
from functools import partial
from engine.utils import get_statement_dict, ProgramInterpreter
from prompts.nlvr import create_prompt
import argparse
import logging
from openai import OpenAI
import random
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visprog Pipeline"
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
        "--debug",
        action="store_true",
        help="Whether to only test on 10 examples",
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
    if args.api_key is None:
        raise ValueError(
            "Need OPENAI API KEY."
        )
    return args

def generate_gpt(client, prompt):
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt = prompt,
        max_tokens=512,
        temperature=0.7,
        top_p=0.5,
        n=1
    )
    res = response.choices[0].text
    return res

def generate_progs(client, marvl_rows, length):
    prompter = partial(create_prompt)
    statement_dict = get_statement_dict(marvl_rows, length)
    unique_ids = [key for key in statement_dict]
    prog_dict = {}
    pbar = tqdm(total=len(unique_ids), position=0, desc="Generating Programs")
    for unique_id in unique_ids:
        try:
            inputs = dict(statement=statement_dict[unique_id])
            prompt = prompter(inputs)
            prog = generate_gpt(client, prompt)
            prog_dict[unique_id] = prog
        except:
            # in case of any errors, re-generate
            try:
                if (unique_id not in prog_dict):
                    inputs = dict(statement=statement_dict[unique_id])
                    prompt = prompter(inputs)
                    prog = generate_gpt(client, prompt)
                    prog_dict[unique_id] = prog
            except Exception as e:
                logging.info(f"An error occurred in generating program for unique_id {unique_id}. Error message: {str(e)}")
        pbar.update(1)
    pbar.close()
    return prog_dict

def execute_progs(marvl_rows, prog_dict, length):
    correct = 0
    err = 0
    output = []
    interpreter = ProgramInterpreter(dataset='nlvr')
    for idx in range(length):
        id = marvl_rows[idx]['id']
        label = marvl_rows[idx]['label'] == "True"
        unique_id = marvl_rows[idx]['unique_id']
        try:
            left_image = Image.open(marvl_rows[idx]['image_1_path'])
            left_image.thumbnail((640,640),Image.Resampling.LANCZOS)
            right_image = Image.open(marvl_rows[idx]['image_2_path'])
            right_image.thumbnail((640,640),Image.Resampling.LANCZOS)
            init_state = dict(
                LEFT=left_image.convert('RGB'),
                RIGHT=right_image.convert('RGB'),
            )
            prog = prog_dict[unique_id]
            result, prog_state, html_str = interpreter.execute(prog, init_state, inspect=True)
            prog_state.pop('LEFT', None)
            prog_state.pop('RIGHT', None)
            if (result != True and result != False): 
                result = random.choice([True, False])
                err += 1
                logging.info(f"Randomly choosing result for idx {idx}")
            if (label == result): correct += 1
            data = {'id': id, 'prediction': result, 'prog': prog, 'prog_state': prog_state, 'html_str': html_str}
            output.append(data)
        except Exception as e:
            logging.info(f"An error occurred in executing program for idx {idx}. Randomly choosing result. Error message: {str(e)}")
            result = random.choice([True, False])
            if (label == result): correct += 1
            data = {'id': id, 'prediction': result, 'prog': '', 'prog_state': '', 'html_str': ''}
            output.append(data)
            err += 1
        logging.info(f"Correct: {correct}, Incorrect: {idx + 1 - err - correct}, Total: {idx + 1}, Accuracy: {correct / (idx + 1)}, Failed: {err}")
    return output

def main():
    args = parse_args()
    logging.info(args)
    client = OpenAI(api_key = args.api_key)
    with open(args.test_file, 'r', newline='') as file:
        reader = csv.DictReader(file)
        marvl_rows = list(reader)
    length = len(marvl_rows)
    if (args.debug):
        length = 5
    prog_dict = generate_progs(client, marvl_rows, length)
    output = execute_progs(marvl_rows, prog_dict, length)
    with open(args.output_file, mode='w', newline='') as file:
        fieldnames = ['id', 'prediction', 'prog', 'prog_state', 'html_str']
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
logging.getLogger('httpx').setLevel(logging.WARNING)

if __name__ == "__main__":
    main()