import os
from pathlib import Path


def get_split(path):
    dataset_path = Path(path)

    train_file_names = []
    val_file_names = []
    test_file_names = []
    sets = os.listdir(dataset_path)
    for set in sets:
        if set is "Train":
            train_file_names = os.listdir(dataset_path / (set) / 'images')
        elif set is "Test":
            test_file_names = os.listdir(dataset_path / (set) / 'images')
        else:
            val_file_names = os.listdir(dataset_path / (set) / 'images')

    return train_file_names, test_file_names, val_file_names
