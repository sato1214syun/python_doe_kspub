# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

# SMILES 付きデータセットの読み込み
dataset = pd.read_csv("test_data/molecules.csv", index_col=0)
smiles = dataset.iloc[:, 0]  # 分子の SMILES
print("分子の数 :", len(smiles))
if dataset.shape[1] > 1:
    y = dataset.iloc[:, 1:]  # 物性・活性などの Y

# 計算する記述子名の取得
descriptor_names = []
for descriptor_information in Descriptors.descList:
    descriptor_names.append(descriptor_information[0])
print("計算する記述子の数 :", len(descriptor_names))

# 記述子の計算
descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
    descriptor_names
)
descriptors = []  # ここに計算された記述子の値を追加
for index, smiles_i in enumerate(smiles):
    print(index + 1, "/", len(smiles))
    molecule = Chem.MolFromSmiles(smiles_i)
    descriptors.append(descriptor_calculator.CalcDescriptors(molecule))
descriptors = pd.DataFrame(descriptors, index=dataset.index, columns=descriptor_names)
if dataset.shape[1] > 1:
    descriptors = pd.concat([y, descriptors], axis=1)  # y と記述子を結合

# 保存
# descriptors_with_y = descriptors_with_y.drop(['MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge'], axis=1)
# csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
descriptors.to_csv("sample/output/05_05/csv_descriptors.csv")
