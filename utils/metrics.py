import os
import pandas as pd
import numpy as np


class counting_metrics():
    def __init__(self, class_num, labels=None, outpath='out'):
        self.class_num = class_num

        if labels is not None:
            if len(labels) != class_num:
                print(
                    f'Error! class numbers {class_num} is not equal to labels {labels}'
                )
            self.labels = labels
        else:
            self.labels = [i for i in range(class_num)]

        cols = self.labels[:]
        cols.insert(0, 'fn')
        self.pd_df = pd.DataFrame(columns=cols)
        self.gt_df = pd.DataFrame(columns=cols)
        self.metrics = pd.DataFrame(columns=self.labels)
        self.outpath = outpath

    def add_result(self, result, gt):
        self.pd_df = self.pd_df.append(result, ignore_index=True)
        self.pd_df.fillna(0, inplace=True)
        self.gt_df = self.gt_df.append(gt, ignore_index=True)
        self.gt_df.fillna(0, inplace=True)

    def calc_metrics(self):

        for ci in self.labels:
            f_cnt = (self.gt_df[ci] != 0).sum()
            c_mae = np.sum(np.abs(self.pd_df[ci] - self.gt_df[ci])) / f_cnt
            c_mse = np.sum(np.power(self.pd_df[ci] - self.gt_df[ci],
                                    2)) / f_cnt
            c_mcount = self.gt_df[ci].sum() / f_cnt
            c_accuracy = (c_mcount - c_mae) / c_mcount

            self.metrics.loc['accuracy', ci] = c_accuracy
            self.metrics.loc['mae', ci] = c_mae
            self.metrics.loc['mse', ci] = c_mse
            self.metrics.loc['rmse', ci] = np.sqrt(self.metrics.loc['mse', ci])
        self.metrics.to_csv(os.path.join(self.outpath, 'metrics_result.csv'))
        self.pd_df.to_csv(os.path.join(self.outpath, 'predict.csv'))
        self.gt_df.to_csv(os.path.join(self.outpath, 'groundtruth.csv'))

if __name__ == '__main__':

    tm = counting_metrics(4)
    tm.add_result({'fn':'f1',0:990,1:1490,2:9800,3:14900}, {'fn':'f1',0:1000,1:1500,2:10000,3:15000})
    tm.add_result({'fn':'f2',0:980,1:1480,2:9900,3:14950}, {'fn':'f2',0:1000,1:1500,2:10000,3:15000})
    tm.add_result({'fn':'f3',0:970,1:1470,2:9950,3:14980}, {'fn':'f3',0:1000,1:1500,2:10000,3:15000})
    tm.add_result({'fn':'f4',0:980,1:1490,2:10420}, {'fn':'f4',0:1000,1:1500,2:10000})
    tm.add_result({'fn':'f4',0:0,1:0,2:9980}, {'fn':'f4',0:0,1:0,2:10000})

    tm.calc_metrics()
    print('\nGroundTruth')
    print(tm.gt_df.to_markdown())
    print('\nPredict')
    print(tm.pd_df.to_markdown())
    print('\nMetrics result')
    print(tm.metrics.to_markdown())

    tmc = counting_metrics(4, ['a', 'b', 'c', 'd'])
    tmc.add_result({'fn':'f1','a':990,'b':1490,'c':9800,'d':14900}, {'fn':'f1','a':1000,'b':1500,'c':10000,'d':15000})
    tmc.add_result({'fn':'f2','a':980,'b':1480,'c':9900,'d':14950}, {'fn':'f2','a':1000,'b':1500,'c':10000,'d':15000})
    tmc.add_result({'fn':'f3','a':970,'b':1470,'c':9950,'d':14980}, {'fn':'f3','a':1000,'b':1500,'c':10000,'d':15000})
    tmc.add_result({'fn':'f4','a':980,'b':1490,'c':10420}, {'fn':'f4','a':1000,'b':1500,'c':10000})
    tmc.add_result({'fn':'f4','a':0,'b':0,'c':9980}, {'fn':'f4','a':0,'b':0,'c':10000})

    tmc.calc_metrics()
    print('\nGroundTruth')
    print(tmc.gt_df.to_markdown())
    print('\nPredict')
    print(tmc.pd_df.to_markdown())
    print('\nMetrics result')
    print(tmc.metrics.to_markdown())
