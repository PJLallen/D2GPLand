from pathlib import Path


def get_split(path):
    train_path = Path(path)

    train_file_names = []
    val_file_names = []
    test_file_names = []
    for instrument_id in range(1, 63):
        if instrument_id in [21, 41, 31, 51]:
            test_file_names += list((train_path / ('Patient_' + str(instrument_id)) / 'images').glob('*'))
        elif instrument_id in [18, 40, 32]:
            val_file_names += list((train_path / ('Patient_' + str(instrument_id)) / 'images').glob('*'))  # 122
        else:
            train_file_names += list((train_path / ('Patient_' + str(instrument_id)) / 'images').glob('*'))  # 109


    return train_file_names, test_file_names, val_file_names
