import os
import json
import re
import torch
import transformers
import random
import itertools
import uuid
from tqdm import tqdm
from time import time
# from fastchat.model import get_conversation_template


def load_testcases(test_file):
    """Load test cases from a jsonl file."""
    with open(test_file, 'r') as json_file:
        json_list = list(json_file)

    test_cases = []
    for test_case in json_list:
        test_case = json.loads(test_case)
        test_cases.append(test_case)

    return test_cases


def test_lines_one_sample(model, tokenizer, test_case, model_name_or_path, use_cache=True):
    """
    Test a single lines task sample.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        test_case: A dictionary containing 'prompt', 'correct_line', 'expected_number', 'random_idx'
        model_name_or_path: Model name or path for template selection
        use_cache: Whether to use cache during generation
    
    Returns:
        tuple: (is_correct, prompt_length, summary)
    """
    prompt = test_case["prompt"]
    correct_line = test_case["correct_line"]
    expected_number = test_case["expected_number"]

    # Use conversation template for chat models
    if "longchat" in model_name_or_path.lower():
        conv = get_conversation_template("vicuna")
    else:
        conv = get_conversation_template(model_name_or_path)

    if "mosaicml/mpt-30b-chat" in model_name_or_path:
        prompt += f'Answer in the format <{test_case["random_idx"][0]}> <REGISTER_CONTENT>.'
    
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input = tokenizer(prompt, return_tensors="pt")
    prompt_length = input.input_ids.shape[-1]
    
    device = getattr(model, "device", "cpu")
    
    # Generate with stopping criteria
    try:
        stopping_criteria = transformers.StopStringCriteria(tokenizer, [".", "###"])
        output = model.generate(
            input.input_ids.to(device), 
            max_new_tokens=100, 
            use_cache=use_cache,
            stopping_criteria=[stopping_criteria]
        )[0]
    except:
        # Fallback if StopStringCriteria is not available
        output = model.generate(
            input.input_ids.to(device), 
            max_new_tokens=100, 
            use_cache=use_cache
        )[0]
    
    output = output[prompt_length:]
    output = tokenizer.batch_decode([output], skip_special_tokens=True)[0]

    # Matching the last digit of the model output
    response_number = re.findall("\d+", output)
    if response_number is not None and len(response_number) > 0:
        response_number = int(response_number[-1])
    else:
        print(f"Got unparsable result: {output}")
        response_number = -1

    is_correct = expected_number == response_number
    summary = f"Label: {expected_number}, Predict: {output}, Parsed: {response_number}, prompt length: {prompt_length}".replace('\n', ' ')
    
    return is_correct, prompt_length, summary


def eval_longeval_lines(model, 
                       tokenizer, 
                       test_dir,
                       model_name_or_path,
                       num_lines_list=None,
                       eval_shortest_only=False,
                       result_path='',
                       use_cache=True):
    """
    Evaluate model on longeval lines task.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        test_dir: Directory containing test cases (should have lines/testcases/ subdirectory)
        model_name_or_path: Model name or path for template selection
        num_lines_list: List of number of lines to test (default: [200, 300, 400, 500, 600, 680, 700, 800, 900, 1000, 1100, 1200, 1350])
        eval_shortest_only: If True, only evaluate the shortest case
        result_path: Path to save results JSON file
        use_cache: Whether to use cache during generation
    
    Returns:
        dict: Results dictionary with accuracy for each number of lines
    """
    if num_lines_list is None:
        num_lines_list = [200, 300, 400, 500, 600, 680, 700, 800, 900, 1000, 1100, 1200, 1350]
    
    results = {}
    start_time = time()
    
    for num_lines in num_lines_list:
        print(f"************ Start testing {num_lines} lines per LRT prompt ************")
        test_file = os.path.join(test_dir, f"lines/testcases/{num_lines}_lines.jsonl")
        
        if not os.path.exists(test_file):
            print(f"Test file not found: {test_file}, skipping...")
            continue
        
        test_cases = load_testcases(test_file)
        num_correct = 0
        avg_length = 0
        
        for idx, test_case in tqdm(enumerate(test_cases), desc=f"Evaluating {num_lines} lines"):
            correct, prompt_length, summary = test_lines_one_sample(
                model=model, 
                tokenizer=tokenizer, 
                test_case=test_case,
                model_name_or_path=model_name_or_path,
                use_cache=use_cache
            )
            avg_length += prompt_length / len(test_cases)
            num_correct += correct
        
        accuracy = num_correct / len(test_cases) if len(test_cases) > 0 else 0.0
        results[f"{num_lines}_lines"] = {
            "accuracy": accuracy,
            "num_correct": num_correct,
            "num_total": len(test_cases),
            "avg_prompt_length": avg_length
        }
        
        print(f"************ Finish testing {num_lines} lines per prompt with average prompt length {avg_length:.2f}, accuracy: {accuracy:.4f} ************")
        
        if eval_shortest_only:
            break
    
    elapsed_time = time() - start_time
    results["total_time"] = elapsed_time
    
    if result_path:
        os.makedirs(os.path.dirname(result_path) if os.path.dirname(result_path) else '.', exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {result_path}")
    
    print(f"LongEval Lines Task Time: {elapsed_time:.2f}s")
    print(f"Results: {results}")
    
    return results


def generate_line_index(num_line, idx_opt):
    """
    Generate line indices based on the specified option.
    
    Args:
        num_line: Number of lines to generate
        idx_opt: Index option ('LRT', 'LRT-ABCindex', 'LRT-UUID', 'LRT-NL')
    
    Returns:
        list: List of line indices
    """
    if idx_opt == "LRT":
        return list(range(1, num_line + 1))
    elif idx_opt == "LRT-ABCindex":
        ingredients = ["A", "B", "C", "D", "E", "F"]
        start = 6
        comb = list(itertools.product(ingredients, repeat=start))
        while len(comb) < num_line:
            start += 1
            comb = list(itertools.product(ingredients, repeat=start))
        comb = ["".join(i) for i in comb]
        return comb[:num_line]
    elif idx_opt == "LRT-UUID":
        comb = []
        for i in range(num_line):
            comb.append(str(uuid.uuid4()))
        return comb
    elif idx_opt == "LRT-NL":
        try:
            import wonderwords
            w = wonderwords.RandomWord()
            adjs = w.random_words(num_line, include_categories=["adjective"])
            nouns = w.random_words(num_line, include_categories=["noun"])
            comb = []
            for i, (adj, noun) in enumerate(zip(adjs, nouns)):
                comb.append(f"{adj}-{noun}")
            return comb
        except ImportError:
            print("Warning: wonderwords not installed, falling back to LRT")
            return list(range(1, num_line + 1))
    else:
        # Default to LRT
        return list(range(1, num_line + 1))


def retrieve_expected(lines, random_line_pos):
    """
    Retrieve the expected number from a line.
    
    Args:
        lines: List of lines
        random_line_pos: Position of the line to retrieve
    
    Returns:
        tuple: (expected_number, correct_line)
    """
    correct_line = lines[random_line_pos]
    expected_number = re.search("<\d+>", correct_line)
    if expected_number is not None:
        expected_number = int(expected_number.group()[1:-1])
    else:
        print(f"Got unparsable line: {correct_line}")
        expected_number = -1
    return expected_number, correct_line


def generate_prompt_from_lines(lines):
    """
    Generate a prompt string from a list of lines.
    
    Args:
        lines: List of lines
    
    Returns:
        str: Combined prompt string
    """
    prompt = ""
    for l in lines:
        prompt += l
    return prompt


def generate_lines_testcases(num_lines_list, num_test_samples, line_idx_opt, output_dir):
    """
    Generate test cases for the lines task.
    
    Args:
        num_lines_list: List of number of lines to generate testcases for
        num_test_samples: Number of test samples per number of lines
        line_idx_opt: Line index option ('LRT', 'LRT-ABCindex', 'LRT-UUID', 'LRT-NL')
        output_dir: Directory to save testcases (should have lines/testcases/ subdirectory)
    """
    testcases_dir = os.path.join(output_dir, "lines", "testcases")
    os.makedirs(testcases_dir, exist_ok=True)
    
    for n in num_lines_list:
        output_path = os.path.join(testcases_dir, f"{n}_lines.jsonl")
        f = open(output_path, "w")
        
        print(f"Generating {num_test_samples} testcases for {n} lines...")
        
        for i in tqdm(range(num_test_samples), desc=f"Generating {n} lines"):
            prompt_header = "Below is a record of lines I want you to remember. " + \
                            "Each line begins with 'line <line index>' and contains " + \
                            "a '<REGISTER_CONTENT>' at the end of the line as a numerical value. " + \
                            "For each line index, memorize its corresponding <REGISTER_CONTENT>. At " + \
                            "the end of the record, I will ask you to retrieve the corresponding " + \
                            "<REGISTER_CONTENT> of a certain line index. Now the record start:\n\n"
            
            lines_list = []
            
            if line_idx_opt == "LRT":
                line_idxes = list(range(1, n + 1))
                lines_list.extend([f"line {i}: REGISTER_CONTENT is <{random.randint(1, 50000)}>\n" for i in line_idxes])
                random_idx = random.randint(1, n)
                random_num = random_idx - 1
            else:
                line_idxes = generate_line_index(n, line_idx_opt)
                lines_list.extend([f"line {i}: REGISTER_CONTENT is <{random.randint(1, 50000)}>\n" for i in line_idxes])
                random_num = random.randint(0, len(line_idxes)-1)
                random_idx = line_idxes[random_num]
            
            expected_number, correct_line = retrieve_expected(lines_list, random_num)
            lines_list.insert(0, f"{prompt_header}")
            lines_list.insert(len(lines_list), f"\nNow the record is over. Tell me what is the <REGISTER_CONTENT> in line {random_idx}? I need the number.")
            prompt = generate_prompt_from_lines(lines_list)
            
            output = {
                "random_idx": (random_idx, random_num),  # this is the line to retrieve
                "expected_number": expected_number,
                "num_lines": n,
                "correct_line": correct_line,
                "prompt": prompt
            }
            
            json.dump(output, f)
            f.write("\n")
        f.close()
        print(f"Saved {num_test_samples} testcases to {output_path}")
