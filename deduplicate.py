import datetime
import difflib
import os
import glob
import hashlib
import re
from typing import Dict, List

p = re.compile("^(\d+)(.+$)")


def list_by_pattern(pattern: str = "*.py", directory: str = "backup") -> List[str]:
    return [os.path.join(directory, f) for f in glob.glob(pattern, root_dir=directory)]


def group_identical(pattern: str = "*.py", directory: str = "backup") -> Dict[str, List[str]]:
    files = list_by_pattern(pattern, directory)
    group_by_hashes = {}
    for file in files:
        with open(file, "r") as f:
            file_hash = hashlib.sha256(f.read().encode()).hexdigest()
            group_by_hashes.setdefault(file_hash, []).append(file)
    return group_by_hashes


def remove_duplicates(pattern: str = "*.py", directory: str = "backup"):
    groups: Dict[str, List[str]] = group_identical(pattern, directory)
    for h, group in groups.items():
        group.sort()
        print(f"Removing duplicates for hash {h}")
        for duplicate in group[1:]:
            os.remove(duplicate)
        rename_remaining(group[0])
    print(f"{len(groups)} distinct versions found")


def epoch_to_readable(epoch: int) -> str:
    return datetime.datetime.utcfromtimestamp(epoch).isoformat()


def rename_remaining(path: str):
    path = os.path.split(path)
    filepath = path[-1]
    timestamp, rest = p.search(filepath).groups()
    timestamp_readable = epoch_to_readable(int(timestamp))
    os.rename(os.path.join(*path), os.path.join(*path[:-1], timestamp_readable + rest))


def diff_files(file1: str, file2: str):
    with open(file1, "r") as f1, open(file2, "r") as f2:
        return "".join(difflib.unified_diff(f1.readlines(), f2.readlines()))


def print_diffs(diff_name, pattern: str = "*.py", directory: str = "backup"):
    files = sorted(list_by_pattern(pattern, directory))
    with open(f"{diff_name}.diff", "w") as f:
        for first, second in zip(files, files[1:]):
            f.write(f"\nFirst file: {first}\nSecond file: {second}\n\nDiff:\n{diff_files(first, second)}\n")


if __name__ == "__main__":
    # Run this from the same directory where you store the backup directory with duplicate files
    remove_duplicates(pattern="*_finetune.py", directory="backup")
    remove_duplicates(pattern="*_config.yaml", directory="backup")
    print_diffs("finetune", pattern="*_finetune.py", directory="backup")
    print_diffs("config", pattern="*_config.yaml", directory="backup")
