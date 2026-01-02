# -*- coding: utf-8 -*-
"""
@author: Hiromasa Kaneko
"""

import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem import BRICS

number_of_generating_structures = 100  # 繰り返し 1 回あたり生成する化学構造の数
number_of_iterations = 10  # 繰り返し回数。(number_of_generating_structures × number_of_iterations) 個の化学構造が生成されます

# 種構造の SMILES のデータセットの読み込み
dataset = pd.read_csv("test_data/molecules.csv", index_col=0)
molecules = [Chem.MolFromSmiles(smiles) for smiles in dataset.iloc[:, 0]]
print("種となる分子の数 :", len(molecules))

# フラグメントへの変換
fragments = set()
for molecule in molecules:
    fragment = BRICS.BRICSDecompose(molecule, minFragmentSize=1)
    fragments.update(fragment)
print("生成されたフラグメントの数 :", len(fragments))

# 化学構造生成
generated_structures = []
for iteration in tqdm(range(number_of_iterations), desc=" outer", position=0):
    generated_structures_all = BRICS.BRICSBuild(
        [Chem.MolFromSmiles(fragment) for fragment in fragments]
    )
    for index, generated_structure in tqdm(enumerate(generated_structures_all, 1), desc=" inner loop", position=1, leave=False):
        generated_structure.UpdatePropertyCache(True)
        generated_structures.append(Chem.MolToSmiles(generated_structure))
        if index >= number_of_generating_structures:
            break
generated_structures = list(set(generated_structures))  # 重複する構造の削除
generated_structures = pd.DataFrame(generated_structures, columns=["SMILES"])
# csv ファイルに保存。同じ名前のファイルがあるときは上書きされますので注意してください
generated_structures.to_csv("sample/output/05_05/generated_structures_brics.csv")
