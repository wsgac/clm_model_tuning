import json
import os
import random


CONFIG = {
    "subdatasets_path": os.path.expanduser("~/.bittensor/data"),
    "shard_size": 1000,
    "output_filename_template": "train-shard.txt"
}


def get_structure_of_files():
    print("Getting structure of files..")
    directories = os.listdir(CONFIG["subdatasets_path"])
    result = {}
    for dir in directories:
        path = CONFIG["subdatasets_path"] + os.sep + dir
        onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        random.shuffle(onlyfiles)  # Effect of randomization
        result[path] = onlyfiles
    return result


def create_shard(queue):
    # Create string from opened hashes
    print("Creating shards from list of hashes..x")

    separator = "||--<<END>>--||"
    file_counter = 0
    shard = ""

    for i, item in enumerate(queue):
        if file_counter == CONFIG["shard_size"]:
            break
        with open(item, "r") as f:
            try:
                text = json.loads(f.read())['Data']
            except Exception:
                print(f"Could not load file: {item}")
                continue
        shard += text
        shard += separator
        file_counter += 1

    return shard


def create_queue(files_dict):
    # Create list of all paths to hashes and shuffle them
    print("Creating list of hashes..")

    result = []
    for subdataset, hashes in files_dict.items():
        for hash in hashes:
            result.append(subdataset + os.sep + hash)
    random.shuffle(result)
    return result


def save_shard_to_file(shard):
    # Ensure file does not exist
    i = 0
    output_filename = str(i) + "-" + CONFIG["output_filename_template"]
    while os.path.exists(output_filename):
        i += 1
        output_filename = str(i) + "-" + CONFIG["output_filename_template"]

    # Save shard to a file
    print("Saving shard to a file..")
    with open(output_filename, "w") as f:
        f.write(shard)


def main():
    files_dict = get_structure_of_files()
    queue = create_queue(files_dict)
    shard = create_shard(queue)
    save_shard_to_file(shard)


if __name__ == "__main__":
    main()
