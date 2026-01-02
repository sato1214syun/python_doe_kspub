# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

import pandas as pd

dataset = pd.read_csv("test_data/resin.csv", index_col=0, header=0)

covariance = dataset.cov()  # 共分散の計算
covariance.to_csv("sample/output/03_03/covariance.csv")

correlation_coefficient = dataset.corr()  # 相関係数の計算
correlation_coefficient.to_csv("sample/output/03_03/correlation_coefficient.csv")