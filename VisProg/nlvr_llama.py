import csv
import logging
from PIL import Image
from functools import partial
from engine.utils import ProgramGenerator, ProgramInterpreter, get_statement_dict
from prompts.nlvr import create_prompt
from text_generation import AsyncClient
import argparse
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
        "--model_addr",
        type=str,
        default=None,
        help="The address of llama model to use",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=4,
        help="The number of data points run in each iteration",
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
    if args.model_addr is None:
        raise ValueError(
            "Need model address."
        )
    return args

def generate_progs(async_client, marvl_rows, length, sample_size):
    prompter = partial(create_prompt)
    generator = ProgramGenerator(prompter=prompter, client=async_client)
    statement_dict = get_statement_dict(marvl_rows, length)
    unique_ids = [key for key in statement_dict]
    prog_dict = {}
    pbar = tqdm(total=len(unique_ids), position=0, desc="Generating Programs")
    # generating programs in batch, with batch size = sample_size
    for i in range(0, len(unique_ids), sample_size):        
        try:
            sample_unique_ids = unique_ids[i : (i + sample_size)]
            inputs_list = [dict(statement=statement_dict[unique_id]) for unique_id in sample_unique_ids]
            progs = generator.generate_async(inputs_list)
            for idx in range(len(sample_unique_ids)):
                prog_dict[sample_unique_ids[idx]] = progs[idx]
        except Exception as e:
            logging.info(f"An error occurred in generating program for unique_id {unique_id}. Error message: {str(e)}")
        pbar.update(1)
    pbar.close()
    # in case of any errors, redo this but not in batch
    for unique_id in unique_ids:
        try:
            if (unique_id not in prog_dict):
                inputs = dict(statement=statement_dict[unique_id])
                prog = generator.generate_async([inputs])
                prog_dict[unique_id] = prog[0]
        except Exception as e:
            logging.info(f"An error occurred in re-generating program for unique_id {unique_id}. Error message: {str(e)}")
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
        logging.info(f"Correct: {correct}, Incorrect: {idx + 1 - err - correct}, Accuracy: {correct / (idx + 1)}, Failed: {err}")
    return output

def main():
    args = parse_args()
    logging.info(args)
    with open(args.test_file, 'r', newline='') as file:
        reader = csv.DictReader(file)
        marvl_rows = list(reader)
    length = len(marvl_rows)
    async_client = AsyncClient(("http://" + args.model_addr))
    if (args.debug):
        length = 5
    prog_dict = generate_progs(async_client, marvl_rows, length, args.sample_size)
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

if __name__ == "__main__":
    main()


