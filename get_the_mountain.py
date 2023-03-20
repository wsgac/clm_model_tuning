import bittensor
import requests
import numpy as np
from multiprocessing.pool import ThreadPool

IPFS_ENDPOINT = "http://global.ipfs.opentensor.ai/"


def request_to_ipfs(hash):
    result = requests.post(IPFS_ENDPOINT + "api/v0/object/get?arg=" + hash)
    if result.status_code == 200:
        return result.json()


def get_hash_table():

    def save_leaf(l, name):
        l['Folder'] = name
        dataset.get_text(l)

    the_mountain_parent_hash = "QmSdDg6V9dgpdAFtActs75Qfc36qJtm9y8a7yrQ1rHm7ZX"
    result = request_to_ipfs(the_mountain_parent_hash)
    files_meta = result['Links']
    dataset = bittensor.dataset(no_tokenizer=True, save_dataset=False)
    dataset.backup_dataset_cap_size = 2e12
    for file_meta in files_meta:
        name = file_meta['Name'].split(".")[0]
        file_meta['Folder'] = name
        leaves = []
        dataset.save_dataset = False
        leaves.extend(dataset.get_dataset(file_meta))
        dataset.save_dataset = True

        with ThreadPool(50) as pool:
            out = pool.map(lambda x: save_leaf(x, name), leaves)


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


def run():
    get_hash_table()


run()
