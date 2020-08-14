import pandas as pd
from utilis import read_json
from utilis.package import *


def main():
    run_fn = "../run"
    dataset = []
    Miou = []
    seq_len = []
    Recall = []
    AP = []
    Recall_time = []
    mAP = []
    Accuracy = []
    Accuracy1 = []
    Accuracy0 = []
    for root, dirs, files in os.walk(run_fn, topdown=False):
        for name in files:
            if name == "log.json":
                print("...found {}".format(osp.join(root, name)))
                log_fn = osp.join(root, name)
                log_dict = read_json(osp.join(log_fn))
                dataset.append(log_dict['cfg']['dataset'])
                seq_len.append(log_dict['cfg']['seq_len'])
                AP.append(log_dict['final']['AP'])
                mAP.append(log_dict['final']['mAP'])
                Accuracy.append(log_dict['final']['Accuracy'])
                Accuracy1.append(log_dict['final']['Accuracy1'])
                Accuracy0.append(log_dict['final']['Accuracy0'])
                Miou.append(log_dict['final']['Miou'])
                Recall.append(log_dict['final']['Recall'])
                Recall_time.append(log_dict['final']['Recall_time'])

    df = pd.DataFrame({
        'dataset': dataset,
        'seq_len': seq_len,
        'AP': AP,
        'mAP': mAP,
        'Miou': Miou,
        'Recall': Recall,
        'Recall_time': Recall_time,
        'Accuracy': Accuracy,
        'Accuracy1': Accuracy1,
        'Accuracy0': Accuracy0
    })
    csv_save_path = osp.join(run_fn, 'all.csv')
    df.to_csv(csv_save_path)
    print("...all results are saved into {}".format(csv_save_path))


if __name__ == '__main__':
    main()
