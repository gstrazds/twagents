from datasets import load_dataset, concatenate_datasets
from pathlib import Path

def load_textds(dirpath, splits_list = None):
    def _normalize_splitname(splitname):
        name_parts = splitname.split('-')
        if 'train' in splitname:
            return 'train'
        elif 'valid' in splitname:
            return 'valid'
        elif 'test' in splitname:
            return 'test'
        return splitname

    if splits_list is None:
        splits_list = ['train', 'valid', 'test']
    dsfiles = {_normalize_splitname(split): str(Path(dirpath, f"{split}.textds")) for split in splits_list}
    print("load_textds:", dsfiles)

    _dataset = load_dataset('json', data_files=dsfiles, download_mode='force_redownload')
    return _dataset


def merge_datasets(twdata_dir='/ssd2tb/twdata', output_dir='/ssd2tb/twdata/combined'):
    ftwc_ds = load_textds(twdata_dir, splits_list=['train', 'valid', 'test'])
    print("FTWC[train][0]:\n", ftwc_ds['train'][0]['source'])
    gata_ds = load_textds(twdata_dir, splits_list=['gata_train', 'gata_valid', 'gata_test'])
    print("GATA[train][0]:\n", gata_ds['train'][0]['source'])
    for splitname in ftwc_ds:
        ftwc_ds[splitname] = concatenate_datasets([ftwc_ds[splitname], gata_ds[splitname]])
    for splitname in ftwc_ds:
        ftwc_ds[splitname].to_json(f'{output_dir}/{splitname}.textds')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge FTWC and GATA playthrough datasets")
    parser.add_argument("--twdata-dir", default='/ssd2tb/twdata', metavar="PATH",
                        help="Path to directory containing .textds files to merge")
    parser.add_argument("--output-dir", default="/ssd2tb/twdata/combined", metavar="PATH",
                        help="Path to directory for merged .textds files")
    args = parser.parse_args()
    merge_datasets(twdata_dir=args.twdata_dir, output_dir=args.output_dir)
