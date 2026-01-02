# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

import pandas as pd

dataset = pd.read_csv("test_data/resin.csv", index_col=0)

statistics = pd.concat(
    [
        dataset.mean(),
        dataset.median(),
        dataset.var(),
        dataset.std(),
        dataset.max(),
        dataset.min(),
        dataset.sum(),
    ],
    axis=1,
).T  # 統計量を計算して結合
statistics.index = [
    "mean",
    "median",
    "variance",
    "standard deviation",
    "max",
    "min",
    "sum",
]
statistics.to_csv("sample/output/03_03/statistics.csv")  # csv ファイルとして保存
