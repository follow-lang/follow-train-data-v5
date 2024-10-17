from huggingface_hub import hf_hub_download
import zipfile
import itertools
import os
import json
import shutil
from huggingface_hub import HfApi
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed


batch_size = 5000
global_vars = set()
word_map: dict[str, int] = {}
max_len = 2048
n_thread = 8
n_futures = 128

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

def s2i(stmt: str): 
    stmt = stmt.strip()
    if len(stmt) == 0:
        return []
    return [word_map.get(word, -1) for word in stmt.split(" ")]


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
        rst.append("diff")
        for dv in dvs:
            rst.append(" ".join(["(", dv[0], ",", dv[1], ")"]))
    rst += tails
    rst.append("<end>")
    return " ".join(rst)


def get_axiom_train_data(axiom, arg_map={}):
    new_targets, new_conditions, new_diffs = stmt_subs(
        axiom["targets"], axiom["conditions"], axiom["dvs"], arg_map
    )
    rst = get_block_train_data(new_targets, new_conditions, new_diffs)
    rst = " ".join([rst, rst, "<qed>", "<eos>"]) # [state, action, <qed>]
    return [rst], []


def get_thm_train_data(thm, arg_map={}):
    new_targets, new_conditions, new_diffs = stmt_subs(
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

    new_states = [get_block_train_data(new_targets, [], [], tails)]

    memories = []
    for idx in range(len(actions)):
        new_state_tokens = new_states[idx]

        a_targets, a_conditions, a_dvs = actions[idx]
        new_a_targets, new_a_conditions, new_a_dvs = stmt_subs(
            a_targets, a_conditions, a_dvs, arg_map
        )
        action_tokens = get_block_train_data(new_a_targets, new_a_conditions, new_a_dvs)

        next_state = states[idx + 1]
        if len(next_state) > 0:
            new_next_state, _, _ = stmt_subs(next_state, [], [], arg_map)
            new_next_state_tokens = get_block_train_data(new_next_state, [], [], tails)
            memories.append(
                " ".join([new_state_tokens, action_tokens, new_next_state_tokens])
            )
            new_states.append(new_next_state_tokens)
        else:
            memories.append(" ".join([new_state_tokens, action_tokens, "<qed>", "<eos>"]))
            new_states.append("")
    new_operators = []
    for op_label, op_args in thm["operators"]:
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

def check_seq(toks: list[int], max_len=max_len):
  if len(toks) < max_len:
    return True
  return False 

def get_deep_seqs(operations, depth=0, max_len=max_len):
  for op_label, op_args in operations:
    try:
      op_data, op_operations = get_train_data(op_label, op_args)
    except Exception as e:
      print(e) 
      continue 
    for stmt in op_data:
      yield s2i(stmt)
    if len(op_operations) > 0 and depth > 0:
      yield from get_deep_seqs(op_operations, depth - 1, max_len) 

def generate_thm(index, thm, folder, depth=0):
    data, operations = get_train_data(thm)
    invalid = False
    for stmt in data:
        if not check_seq(s2i(stmt)):
            invalid = True  # 至少当前定理证明过程应该满足完整性
            break
    if invalid:
        return 
    valid_seq_f = open(os.path.join(folder, thm + '.txt'), "w") 
    for stmt in data:
        seq = s2i(stmt)
        valid_seq_f.write(' '.join([str(i) for i in seq]) + "\n")
    for seq in get_deep_seqs(operations, depth, max_len):
        if not check_seq(seq):
            continue
        valid_seq_f.write(' '.join([str(i) for i in seq]) + "\n")
    valid_seq_f.close()
    print(f"{index}: {thm}")


def generate_thms(start_idx: int, end_idx:int, train_dir: str, depth=0):
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    os.makedirs(train_dir)

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
            futures.append(executor.submit(generate_thm, index, thm, train_dir, depth))
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
        for file_path in tqdm(file_list, desc="压缩中", unit="文件"):  # 添加进度条
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

    types = read_config("types.txt")
    terms = read_config("terms.txt")
    axioms = read_config("axioms.txt")
    thms = read_config("thms.txt")
    words = read_config("words.txt")

    for t in ["wff", "setvar", "class"]:
        for idx in range(200):
            global_vars.add(f"g{t[0]}{idx}")
            global_vars.add(f"v{t[0]}{idx}")
    for idx, word in enumerate(words):
        word_map[word] = idx
    
    upload('databases/words.txt') # 上传单词表 
    
    n_thms = 20000 # github 只能支持到20000 # len(thms) # 测试5000条

    for start_idx in range(0, n_thms, batch_size):
        end_idx = start_idx + batch_size if start_idx + batch_size < n_thms else n_thms
        train_dir = f'databases/train_{start_idx}_{end_idx-1}'
        output_zip = train_dir + ".zip" 
        if start_idx == 0:
            generate_thms(start_idx, end_idx, train_dir, 3) # 第一部分的数据可以追溯到depth=3
        else:
            generate_thms(start_idx, end_idx, train_dir, 1)
        zip_dataset(train_dir, output_zip)
        upload(output_zip)
        shutil.rmtree(train_dir)
        os.remove(output_zip)
