import os
import json
import pandas as pd


osp = os.path


def main():
    eval_root = './eval'
    eval_folder = osp.join(eval_root, '2020_10_28')
    # eval_folder = osp.join(eval_root, '2020_11_04')

    data = dict()
    for p in os.walk(eval_folder):
        pth, dirs, files = p

        folder = osp.basename(pth)
        if len(files) == 0:
            continue
        if not any([f.endswith('json') for f in files]):
            continue

        print(f'parsing "{folder}"')
        with open(osp.join(pth, sorted(files)[-1])) as fp:
            stats = json.load(fp)

        columns = list(stats.keys())
        data[folder] = stats

    print(f'saving csv')
    df = pd.DataFrame.from_dict(data, orient='index', columns=columns)
    df.to_csv(osp.join(eval_folder, 'summary.csv'))


if __name__ == '__main__':
    main()