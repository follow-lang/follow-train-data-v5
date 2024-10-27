from huggingface_hub import hf_hub_download
import zipfile
import itertools
import os
import json
import shutil
from huggingface_hub import HfApi
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

import threading

total_memory_file_number = 100

write_locks = [threading.Lock() for _ in range(total_memory_file_number)]

global_vars = set()
max_len = 2*1024
n_thread = 32
n_futures = 32
total_memory_count = 0 
max_memory_size = 2*1024*1024
max_depth = 1 # 初始的thm尝试探索深一些
min_thm_number = 40000
max_thm_number = -1
zip_offset = 400

def get_folder_size(folder_path):
    total_size = 0
    # os.walk() generates the file names in a directory tree
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            # Join the directory path with the filename to get full file path
            file_path = os.path.join(dirpath, filename)
            # Only add file size if it's a file (skip if it's a symbolic link, etc.)
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path) / (1024 * 1024) # Convert bytes to MB
    return total_size

def read_config(name, base = "databases"):
  with open(os.path.join(base, name), "r") as f:
    content = [line.strip() for line in f.readlines()]  # 使用 strip() 去掉每行的换行符
  return content

def read_json(name, base = "databases/json"):
  with open(os.path.join(base, name+".json"), "r") as f:
    block = json.load(f)
  return block

def tokenizer(stmt: str): 
    stmt = stmt.strip()
    if len(stmt) == 0:
        return []
    # 减少token的数量 
    toks = [word for word in stmt.split(" ") if word not in ('(', ')', ',')]
    return toks

def stmt_subs(targets, conditions, dvs, arg_map={}):
    new_targets = [
        " ".join([arg_map.get(word, word) for word in ehyp.split(" ")])
        for ehyp in targets
    ]
    new_conditions = [
        " ".join([arg_map.get(word, word) for word in ehyp.split(" ")])
        for ehyp in conditions
    ]
    new_diffs = set()
    if len(dvs) > 0:
        arg_value_map = {
            k: set([word for word in expr.split(" ") if word in global_vars])
            for k, expr in arg_map.items()
        }
        for v1, v2 in dvs:
            v1set = arg_value_map.get(v1, [v1])
            v2set = arg_value_map.get(v2, [v2])
            for x, y in itertools.product(v1set, v2set):
                new_diffs.add((min(x, y), max(x, y)))
    return new_targets, new_conditions, new_diffs

def get_block_train_data(targets, conditions, dvs, tails=[]):
    rst = []
    for target in targets:
        rst.append("|- " + target)
    for condition in conditions:
        rst.append("-| " + condition)
    if dvs and len(dvs) > 0:
        for dv in dvs:
            rst.append(" ".join(["diff (", dv[0], ",", dv[1], ")"]))
    rst += tails
    return " ".join(rst)


def get_axiom_train_data(axiom, arg_map={}):
    new_targets, new_conditions, new_diffs = stmt_subs(
        axiom["targets"], axiom["conditions"], axiom["dvs"], arg_map
    )
    rst = get_block_train_data(new_targets, new_conditions, new_diffs)
    rst = " ".join(["<state>", rst, "</state>", "<action>", rst, "</action>", "<qed>"]) # [state, action, <qed>]
    return [(tokenizer(rst), (1, 1, 0))], [] # (memory, (V(s), V(a), V(s')))


def get_thm_train_data(thm, arg_map={}):
    global total_memory_count, max_memory_size

    _, new_conditions, new_diffs = stmt_subs(
        thm["targets"], thm["conditions"], thm["dvs"], arg_map
    )
    tails = []
    for condition in new_conditions:
        tails.append("-| " + condition)

    if len(new_diffs) > 0:
        tails.append("diff")
        for dv in new_diffs:
            tails.append(f"( {dv[0]} , {dv[1]} )")

    states = thm["states"]
    actions = thm["actions"]
    state_costs = thm["state_costs"]
    action_costs = thm["action_costs"]

    new_state_tokens_list = []
    for state in states:
        if len(state) == 0:
            new_state_tokens = '<qed>'
        else:
            new_state, _, _ = stmt_subs(state, [], [], arg_map)
            new_state_tokens = "<state> " + get_block_train_data(new_state, [], [], tails) + " </state>"
        new_state_tokens_list.append(tokenizer(new_state_tokens))
    
    new_action_tokens_list = []
    for a_targets, a_conditions, a_dvs in actions:
        new_a_targets, new_a_conditions, new_a_dvs = stmt_subs(
            a_targets, a_conditions, a_dvs, arg_map
        )
        action_tokens = "<action> " + get_block_train_data(new_a_targets, new_a_conditions, new_a_dvs) + " </action>"
        new_action_tokens_list.append(tokenizer(action_tokens))

    memories = []
    for start_idx in range(len(new_action_tokens_list)):
        memory = new_state_tokens_list[start_idx] + new_action_tokens_list[start_idx] + new_state_tokens_list[start_idx+1]
        if len(memory) > max_len:
            continue
        costs = (state_costs[start_idx], action_costs[start_idx], state_costs[start_idx + 1])
        memories.append((memory, costs))
        if total_memory_count + len(memories) >= max_memory_size:
            break
    new_operators = []
    for op_label, op_args in thm["operators"]:
        if total_memory_count + len(memories) + len(new_operators) >= max_memory_size:
            break
        new_op_args = stmt_subs(op_args, [], [], arg_map)[0]
        new_operators.append((op_label, new_op_args))
    return memories, new_operators

def get_train_data(label, input_args=[]):
    block = read_json(label)
    arg_map: dict[str, str] = {}
    if block["type"] not in ["axiom", "thm"]:
        return []
    for a_input, (_, a_name) in zip(input_args, block["args"]):
        arg_map[a_name] = a_input
    if block["type"] == "axiom":
        return get_axiom_train_data(block, arg_map) 
    return get_thm_train_data(block, arg_map) # (memories, new_operators)

def check_seq(memory, max_len=max_len):
    seq = memory[0]
    end_of_action = seq.index('</action>')
    if end_of_action < max_len:
        return True
    return False 

def get_deep_memory(operations, depth=0, max_len=max_len):
    global total_memory_count, max_memory_size
    if total_memory_count >= max_memory_size:
        return
    next_level_operations = []
    for op_label, op_args in operations:
        try:
            op_memories, op_operations = get_train_data(op_label, op_args)
            for memory in op_memories:
                if not check_seq(memory):
                    continue
                total_memory_count += 1
                yield memory
                if total_memory_count >= max_memory_size:
                    break
            if total_memory_count >= max_memory_size:
                break
            next_level_operations.extend(op_operations)
        except Exception as e:
            print(e) 
            continue 
    
    # BFS, 保证deep完整 
    if len(next_level_operations) > 0 and depth > 0 and total_memory_count < max_memory_size:
        yield from get_deep_memory(next_level_operations, depth - 1, max_len) 

def write_memory(memory, folder, zip_index):
    # random write
    file_idx = np.random.randint(0, total_memory_file_number)
    s = ' '.join(memory[0])
    costs = ' '.join([str(i) for i in memory[1]])
    line = costs + '\t' + s + '\n'
    # 使用对应的锁来保护写入操作
    with write_locks[file_idx]:  # 选择相应的锁
        with open(os.path.join(folder, f'{zip_index}-{file_idx}.txt'), "a") as f:
            f.write(line)

def generate_thm(index, thm, folder, depth=0, zip_index=0):
    global total_memory_count
    memories, operations = get_train_data(thm)
    for memory in memories:
        if check_seq(memory):
            write_memory(memory, folder, zip_index)
            total_memory_count += 1
    for memory in get_deep_memory(operations, depth, max_len):
        write_memory(memory, folder, zip_index)
    print(f"{index}: {thm}")


def generate_thms(start_idx: int, end_idx:int, train_dir: str, depth=0, zip_index=0):
    index = start_idx
    # 创建线程池
    with ThreadPoolExecutor(max_workers=n_thread) as executor:
        futures = []
        while index < end_idx:
            if len(futures) >= n_futures:
                # 等待一半的任务完成，释放资源
                while len(futures) < n_futures // 2:
                    for future in as_completed(futures):
                        futures.remove(future)
            thm = thms[index]
            # 提交任务到线程池
            futures.append(executor.submit(generate_thm, index, thm, train_dir, depth, zip_index))
            index += 1
        # 确保所有任务完成
        for future in as_completed(futures):
            future.result()

def zip_dataset(dataset_dir, output_zip):
    file_list = []  # 创建文件列表
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)  # 收集文件路径

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in tqdm(file_list, desc=f"{output_zip}-压缩中", unit="文件"):  # 添加进度条
            zipf.write(file_path, os.path.relpath(file_path, dataset_dir))

def upload(output_zip):
    # 上传数据集到 Hugging Face
    api = HfApi()
    repo_id = "Follow-Lang/set.mm"
    file_name = os.path.basename(output_zip)
    path_in_repo = f"datasets/train/{file_name}"

    # 通过 upload_with_progress 进行直接上传
    with open(output_zip, "rb") as f:  # 以二进制模式打开文件
        # 进行上传
        try:
            print("开始上传到Hugging Face")
            api.upload_file(
                path_or_fileobj=f,  # 传递文件对象
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
            )
            print("上传成功")  # 上传成功提示
        except Exception as e:
            print(f"上传失败: {e}")

def run(start, end, depth, batch_size=128):
    global total_memory_count, max_memory_size
    total_memory_count = 0
    file_index = zip_offset
    train_dir = f'databases/train_{file_index}'
    
    try:
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        os.makedirs(train_dir)

        for start_idx in range(start, end, batch_size):
            end_idx = start_idx + batch_size if start_idx + batch_size < end else end
            print(f"Generating theorems from {start_idx} to {end_idx}...")
            generate_thms(start_idx, end_idx, train_dir, depth, file_index) 

            # 检查文件夹大小
            if total_memory_count > max_memory_size: 
                output_zip = train_dir + ".zip" 
                zip_dataset(train_dir, output_zip)
                upload(output_zip)
                shutil.rmtree(train_dir)
                os.remove(output_zip)
                file_index += 1
                train_dir = f'databases/train_{file_index}'
                total_memory_count = 0
                if os.path.exists(train_dir):
                    shutil.rmtree(train_dir)
                os.makedirs(train_dir)
        
        if os.path.exists(train_dir) and os.listdir(train_dir):  # 检查文件夹是否存在且非空
            output_zip = train_dir + ".zip"
            zip_dataset(train_dir, output_zip)
            upload(output_zip)
            shutil.rmtree(train_dir)
            os.remove(output_zip)
    except Exception as e:
        print(f"运行过程中发生错误: {e}")

if __name__ == "__main__":
    # 删除旧文件夹
    if os.path.exists('databases'):
        shutil.rmtree('databases')

    # 下载数据集并且解压到databases文件夹
    dataset_path = hf_hub_download(repo_id="Follow-Lang/set.mm", repo_type="dataset", filename="datasets/set.mm.zip")
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall("databases/")

    extracted_files = os.listdir("databases/")
    print("Extracted files: ", extracted_files)

    json_files = os.listdir("databases/json")
    print("files: ", len(json_files))
    with open("databases/json/content.follow.json", "r") as config_f:
        config = json.load(config_f)
    file_deps = config["content"]
    print("file_deps: ", len(file_deps))
    # 预期文件夹files中的文件数比 content.follow.json 中记录的文件多1个

    json_size = get_folder_size("databases/json")
    code_size = get_folder_size("databases/code")

    print(f"Total json folder size: {json_size / 1024} GB")
    print(f"Total code folder size: {code_size} MB")

    thms = read_config("thms.txt")
    words = read_config("words.txt")

    for t in ["wff", "setvar", "class"]:
        for idx in range(200):
            global_vars.add(f"g{t[0]}{idx}")
            global_vars.add(f"v{t[0]}{idx}")
    
    for new_word in ['<state>', '</state>', '<action>', '</action>', '<qed>']:
        if new_word not in words:
            words.append(new_word)
    with open("databases/words.txt", 'w') as f:
        f.writelines([word + '\n' for word in words])
    upload("databases/words.txt")

    if max_thm_number < 0:
        max_thm_number = len(thms)
    
    run(min_thm_number, max_thm_number, depth=max_depth, batch_size=n_futures)

