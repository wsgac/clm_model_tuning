import argparse

import bittensor
import requests
import numpy as np
from multiprocessing.pool import ThreadPool

IPFS_ENDPOINT = "http://global.ipfs.opentensor.ai/"
SUBDATASETS = [
    "ArXiv",
    "BookCorpus2",
    "Books3",
    "DMMathematics",
    "EnronEmails",
    "EuroParl",
    "Gutenberg_PG",
    "HackerNews",
    "NIHExPorter",
    "OpenSubtitles",
    "PhilPapers",
    "UbuntuIRC",
    "YoutubeSubtitles"
]


def request_to_ipfs(hash):
    result = requests.post(IPFS_ENDPOINT + "api/v0/object/get?arg=" + hash)
    if result.status_code == 200:
        return result.json()


def create_threading_pool(ds):
    dataset = ds['dataset']
    name = ds['name']
    leaves = ds['leaves']
    with ThreadPool(200) as pool:
        out = pool.map(lambda x: save_leaf(dataset, x, name), leaves)


def save_leaf(dataset, l, name):
    l['Folder'] = name
    dataset.get_text(l)


def run(subdatasets):
    the_mountain_parent_hash = "QmSdDg6V9dgpdAFtActs75Qfc36qJtm9y8a7yrQ1rHm7ZX"
    result = request_to_ipfs(the_mountain_parent_hash)
    files_meta = result['Links']
    dataset = bittensor.dataset(no_tokenizer=True, save_dataset=False)
    dataset.backup_dataset_cap_size = 2e12
    datasets_list = []
    for i, file_meta in enumerate(files_meta):
        name = file_meta['Name'].split(".")[0]
        if name in subdatasets:
            res = {}
            res['name'] = name
            file_meta['Folder'] = name
            leaves = []
            dataset.save_dataset = False
            leaves.extend(dataset.get_dataset(file_meta))
            dataset.save_dataset = True
            res['leaves'] = leaves
            res['dataset'] = dataset
            datasets_list.append(res)

    for ds in datasets_list:
        create_threading_pool(ds)


def random_sieve(data, fraction):
    """
    Sparsify data by keeping a fraction of input data

    :param data: list of  objects
    :param fraction: percentage of data to keep (size-wise)
    :return: filtered list of objects
    """
    total_size = sum(item["Size"] for item in data)
    required_size = fraction * total_size
    result = []
    for i in np.random.permutation(len(data)):
        result.append(data[i])
        required_size -= data[i]["Size"]
        if required_size <= 0:
            break
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subdataset", type=str, default="all")
    args = parser.parse_args()
    if args.subdataset == "all":
        subdatasets = SUBDATASETS
    else:
        subdatasets = [args.subdataset]
    run(subdatasets)
