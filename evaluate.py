import copy
import json
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
import re
import sys
import argparse
from pathlib import Path
import fire

import torch

sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
        load_8bit: bool = False,
        base_model: str = "",
        lora_weights: str = "tloen/alpaca-lora-7b",
        share_gradio: bool = False,
):
    args = parse_args()

    def evaluate(
            instruction,
            input=None,
            temperature=None,
            top_p=None,
            top_k=None,
            num_beams=None,
            max_new_tokens=None,
            do_sample=None,
            use_cache=None,
            **kwargs,
    ):
        temperature = args.temperature if temperature is None else temperature
        top_p = args.top_p if top_p is None else top_p
        top_k = args.top_k if top_k is None else top_k
        num_beams = args.num_beams if num_beams is None else num_beams
        max_new_tokens = args.max_new_tokens if max_new_tokens is None else max_new_tokens
        do_sample = args.do_sample if do_sample is None else do_sample
        use_cache = args.use_cache if use_cache is None else use_cache
        prompt = generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_options = {
            "num_beams": num_beams,
            "do_sample": do_sample,
            **kwargs,
        }
        if do_sample:
            generation_options.update(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        generation_config = GenerationConfig(**generation_options)
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return output.split("### Response:")[1].strip()

    """
    # testing code for readme
    for instruction in [
        "Tell me about alpacas.",
        "Tell me about the president of Mexico in 2019.",
        "Tell me about the king of France in 2019.",
        "List all Canadian provinces in alphabetical order.",
        "Write a Python program that prints the first 10 Fibonacci numbers.",
        "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",  # noqa: E501
        "Tell me five words that rhyme with 'shock'.",
        "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        "Count up from 1 to 500.",
    ]:
        print("Instruction:", instruction)
        print("Response:", evaluate(instruction))
        print()
    """
    save_file = args.output_file or f'experiment/{args.model}-{args.adapter}-{args.dataset}.json'
    output_dir = os.path.dirname(os.path.abspath(save_file))
    os.makedirs(output_dir, exist_ok=True)
    progress_file = save_file + '.jsonl'

    dataset = load_data(args)
    tokenizer, model = load_model(args)
    total = len(dataset)
    completed = {}
    if args.resume and os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as progress:
            for line in progress:
                try:
                    row = json.loads(line)
                    row_index = int(row['index'])
                    if 0 <= row_index < total:
                        completed[row_index] = row
                except (ValueError, KeyError, json.JSONDecodeError):
                    # Ignore an incomplete/corrupt final record after a crash.
                    continue
    else:
        with open(progress_file, 'w', encoding='utf-8'):
            pass

    correct = sum(bool(row.get('flag')) for row in completed.values())
    miss = 0.001
    pbar = tqdm(total=total, initial=min(len(completed), total))
    for idx, data in enumerate(dataset):
        if idx in completed:
            continue
        instruction = data.get('instruction')

        outputs = evaluate(instruction)
        label = data.get('answer')
        flag = False
        if args.dataset.lower() in ['aqua']:
            predict = extract_answer_letter(args, outputs)
            if label == predict:
                correct += 1
                flag = True
        else:
            if isinstance(label, str):
                label = float(label)
            predict = extract_answer_number(args, outputs)
            if abs(label - predict) <= miss:
                correct += 1
                flag = True
        new_data = copy.deepcopy(data)
        new_data['index'] = idx
        new_data['output_pred'] = outputs
        new_data['pred'] = predict
        new_data['flag'] = flag
        completed[idx] = new_data
        print(' ')
        print('---------------')
        print(outputs)
        print('prediction:', predict)
        print('label:', label)
        print('---------------')
        finished = len(completed)
        print(f'\rtest:{finished}/{total} | accuracy {correct}  {correct / max(finished, 1)}')
        with open(progress_file, 'a', encoding='utf-8') as progress:
            progress.write(json.dumps(new_data, ensure_ascii=False) + '\n')
            progress.flush()
        pbar.update(1)
    pbar.close()
    output_data = [completed[index] for index in sorted(completed)]
    temp_file = save_file + '.tmp'
    with open(temp_file, 'w', encoding='utf-8') as result_file:
        json.dump(output_data, result_file, indent=4, ensure_ascii=False)
    os.replace(temp_file, save_file)
    print('\n')
    print('test finished')


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                """  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """  # noqa: E501


def load_data(args) -> list:
    dataset_root = Path(__file__).resolve().parent / "dataset"
    dataset_dir = dataset_root / args.dataset
    # Linux 区分大小写；若精确目录名不存在，尝试忽略大小写匹配。
    if not dataset_dir.is_dir() and dataset_root.is_dir():
        matches = [
            path for path in dataset_root.iterdir()
            if path.is_dir() and path.name.casefold() == args.dataset.casefold()
        ]
        if len(matches) == 1:
            dataset_dir = matches[0]
        elif len(matches) > 1:
            raise RuntimeError(
                f"dataset name {args.dataset!r} matches multiple directories: {matches}"
            )

    file_path = dataset_dir / "test.json"
    if not file_path.is_file():
        raise FileNotFoundError(f"can not find dataset file: {file_path}")

    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['AddSub', 'MultiArith', 'SingleEq', 'gsm8k', 'AQuA', 'SVAMP', 'Test_set'],
                        required=True)
    parser.add_argument('--model', choices=['LLaMA-7B', 'OpenLLaMA-3B', 'TinyLlama-1.1B', 'BLOOM-7B', 'GPT-j-6B'], default="")
    parser.add_argument('--adapter', choices=['LoRA', 'AdapterP', 'AdapterH', 'Parallel', 'Prefix'],
                        required=True)
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--lora_weights', required=True)
    parser.add_argument('--load_8bit', action='store_true', default=False)
    parser.add_argument('--output_file', default='', help='Path for final JSON results')
    parser.add_argument('--resume', action='store_true', help='Resume from the JSONL progress file')
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--top_p', type=float, default=0.75)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--do_sample', action='store_true', default=False)
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument('--use_cache', dest='use_cache', action='store_true')
    cache_group.add_argument('--no_use_cache', dest='use_cache', action='store_false')
    parser.set_defaults(use_cache=True)

    return parser.parse_args()


def load_model(args) -> tuple:
    """
    load tuned model
    Args:
        args:

    Returns:
        tuple(tokenizer, model)
    """
    base_model = args.base_model
    if not base_model:
        raise ValueError(f'can not find base model name by the value: {args.model}')
    if not os.path.isdir(base_model):
        raise FileNotFoundError(
            f'Offline mode requires --base_model to be a local model directory: {base_model}'
        )
    lora_weights = args.lora_weights
    if not lora_weights:
        raise ValueError(f'can not find lora weight, the value is: {lora_weights}')
    if not os.path.isdir(lora_weights):
        raise FileNotFoundError(
            f'Offline mode requires --lora_weights to be a local directory: {lora_weights}'
        )

    load_8bit = args.load_8bit
    if args.model in ('LLaMA-7B', 'OpenLLaMA-3B', 'TinyLlama-1.1B'):
        tokenizer = LlamaTokenizer.from_pretrained(base_model, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        ) # fix zwq
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={"":0}
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True, local_files_only=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

        # unwind broken decapoda-research config
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

    return tokenizer, model


def load_instruction(args) -> str:
    instruction = ''
    if not instruction:
        raise ValueError('instruct not initialized')
    return instruction


def extract_answer_number(args, sentence: str) -> float:
    dataset = args.dataset.lower()
    if dataset in ["multiarith", "addsub", "singleeq", "gsm8k", "svamp", "test_set"]:
        sentence = sentence.replace(',', '')
        pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
        if not pred:
            return float('inf')
        pred_answer = float(pred[-1])
    else:
        raise NotImplementedError(' not support dataset: {}'.format(dataset))
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer


def extract_answer_letter(args, sentence: str) -> str:
    sentence_ = sentence.strip()
    pred_answers = re.findall(r'A|B|C|D|E', sentence_)
    if pred_answers:
        if not pred_answers:
            return ''
        return pred_answers[0]
    else:
        return ''


if __name__ == "__main__":
    main()
