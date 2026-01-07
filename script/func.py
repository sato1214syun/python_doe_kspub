import math
from pathlib import Path
from typing import Literal, overload

import numpy as np
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    Matern,
    Sum,
    WhiteKernel,
)
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_predict,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM


# カーネル 11 種類
def generate_kernels(x: np.ndarray | pl.DataFrame) -> list[Sum]:
    return [
        ConstantKernel() * DotProduct() + WhiteKernel(),
        ConstantKernel() * RBF() + WhiteKernel(),
        ConstantKernel() * RBF() + WhiteKernel() + ConstantKernel() * DotProduct(),
        ConstantKernel() * RBF(np.ones(x.shape[1])) + WhiteKernel(),
        ConstantKernel() * RBF(np.ones(x.shape[1]))
        + WhiteKernel()
        + ConstantKernel() * DotProduct(),
        ConstantKernel() * Matern(nu=1.5) + WhiteKernel(),
        ConstantKernel() * Matern(nu=1.5)
        + WhiteKernel()
        + ConstantKernel() * DotProduct(),
        ConstantKernel() * Matern(nu=0.5) + WhiteKernel(),
        ConstantKernel() * Matern(nu=0.5)
        + WhiteKernel()
        + ConstantKernel() * DotProduct(),
        ConstantKernel() * Matern(nu=2.5) + WhiteKernel(),
        ConstantKernel() * Matern(nu=2.5)
        + WhiteKernel()
        + ConstantKernel() * DotProduct(),
    ]


@overload
def load_data(
    file_path: Path | str, index: int | str
) -> tuple[pl.DataFrame, pl.Series]: ...


@overload
def load_data(file_path: Path | str, index: None = None) -> pl.DataFrame: ...


def load_data(file_path, index=None):
    """Load data from a CSV file."""
    dataset = pl.read_csv(file_path)
    if index is None:
        return dataset
    if isinstance(index, int):
        index = dataset.columns[index]
    y = dataset.get_column(index)
    x = dataset.drop(y.name)
    return x, y


def save_csv(
    data: pl.DataFrame,
    file_path: Path | str,
    add_row_index: bool = False,
    index_name: str = "",
    offset: int = 0,
    quote_style: str = "never",
) -> None:
    """Save DataFrame to a CSV file."""
    if add_row_index:
        data = data.clone().with_row_index(name=index_name, offset=offset)
    data.write_csv(file_path, quote_style=quote_style)


def delete_zero_std_columns(data: pl.DataFrame) -> pl.DataFrame:
    zero_stdev_cols = [
        col for col, std in data.std().row(0, named=True).items() if std in [0, None]
    ]
    data_wo_zero_std = data.drop(zero_stdev_cols)
    return data_wo_zero_std


@overload
def autoscaling(
    data: pl.DataFrame, data2: pl.DataFrame | None = None
) -> pl.DataFrame: ...


@overload
def autoscaling(data: pl.Series, data2: pl.Series | None = None) -> pl.Series: ...


def autoscaling(data, data2=None):
    if data2 is None:
        if isinstance(data, pl.Series):
            return (data - data.mean()) / data.std()
        elif isinstance(data, pl.DataFrame):
            return data.select((pl.all() - pl.all().mean()) / pl.all().std())
        else:
            raise TypeError("Input must be a Polars Series or DataFrame.")

    if isinstance(data, pl.Series) and isinstance(data2, pl.Series):
        return (data - data2.mean()) / data2.std()
    elif isinstance(data, pl.DataFrame) and isinstance(data2, pl.DataFrame):
        return (
            data - data2.mean().select(pl.all().repeat_by(data.height).explode())
        ) / data2.std().select(pl.all().repeat_by(data.height).explode())
    else:
        raise TypeError("Input must be a Polars Series or DataFrame.")


@overload
def rescaling(
    autoscaled_data: pl.DataFrame, original_data: pl.DataFrame, is_std: bool = False
) -> pl.DataFrame: ...


@overload
def rescaling(
    autoscaled_data: pl.Series, original_data: pl.Series, is_std: bool = False
) -> pl.Series: ...


def rescaling(autoscaled_data, original_data, is_std=False):
    if isinstance(autoscaled_data, pl.Series) and isinstance(original_data, pl.Series):
        if is_std:
            return autoscaled_data * original_data.std()
        return autoscaled_data * original_data.std() + original_data.mean()
    elif isinstance(autoscaled_data, pl.DataFrame) and isinstance(
        original_data, pl.DataFrame
    ):
        if is_std:
            return autoscaled_data * original_data.std().select(
                pl.all().repeat_by(autoscaled_data.height).explode()
            )
        return autoscaled_data * original_data.std().select(
            pl.all().repeat_by(autoscaled_data.height).explode()
        ) + original_data.mean().select(
            pl.all().repeat_by(autoscaled_data.height).explode()
        )
    else:
        raise TypeError("Input must be a Polars Series or DataFrame.")


def scatter_plot_of_result(
    save_path: Path | str,
    x: pl.DataFrame,
    y: pl.Series,
    x_label: str = "x",
    y_label: str = "y",
) -> None:
    """実測値 vs. 推定値の散布図を作成して保存する関数"""
    # 実測値 vs. 推定値のプロット
    ax = sns.scatterplot(x=x, y=y, color="blue")
    # 実測値と推定値の両方の最大値・最小値
    concat_data = pl.concat([x, y])
    y_max, y_min = concat_data.max(), concat_data.min()
    # 取得した最小値-5%から最大値+5%まで、対角線を作成
    y_scope_min = y_min - 0.05 * (y_max - y_min)
    y_scope_max = y_max + 0.05 * (y_max - y_min)
    ax.plot([y_scope_min, y_scope_max], [y_scope_min, y_scope_max], "k-")

    ax.set_ylim(y_scope_min, y_scope_max)  # y 軸の範囲の設定
    ax.set_xlim(y_scope_min, y_scope_max)  # x 軸の範囲の設定
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.gca().set_aspect("equal", adjustable="box")  # 図の形を正方形に
    plt.savefig(save_path)
    plt.show()


def show_fitting_evaluation(true_val, estimated_val) -> None:
    """回帰モデルの適合度を評価する関数"""
    # トレーニングデータのr2, RMSE, MAE
    print("r^2 for training data :", r2_score(true_val, estimated_val))
    print(
        "RMSE for training data :",
        root_mean_squared_error(true_val, estimated_val),
    )
    print("MAE for training data :", mean_absolute_error(true_val, estimated_val))


def calc_r2(
    model,
    x: pl.DataFrame,
    y: pl.Series,
    cv: KFold = None,
    y_true: pl.Series = None,
    **model_kwargs,
) -> float:
    if cv is None:
        if y_true is not None:
            return r2_score(
                y_true,
                rescaling(
                    pl.Series(model(**model_kwargs).fit(x, y).predict(x)), y_true
                ),
            )
        return r2_score(y, model(**model_kwargs).fit(x, y).predict(x))

    if y_true is not None:
        return r2_score(
            y_true,
            rescaling(
                pl.Series(cross_val_predict(model(**model_kwargs), x, y, cv=cv)), y_true
            ),
        )
    return r2_score(y, cross_val_predict(model(**model_kwargs), x, y, cv=cv))


def calc_optimal_gamma(x: pl.DataFrame, gammas: pl.Series) -> float:
    """グラム行列の分散が最大となるγを探索する関数"""
    x_wt_var_list = x.select(pl.concat_list(pl.all()).alias("sample"))
    sample_distances = (
        # すべてのサンプルを組み合わせる
        x_wt_var_list.join(x_wt_var_list, how="cross", suffix="_right")
        .select(
            # 全サンプル間の全データのユークリッド距離の二乗を計算
            (pl.col("sample") - pl.col("sample_right"))
            .list.eval(pl.element().pow(2))
            .list.sum()
            .implode()  # グラム行列の計算を簡単にするために、全距離をリスト化
            .alias("distance"),
        )
        .to_series(0)
    )

    gram_matrix = (-gammas * sample_distances).list.eval(pl.element().exp())
    return gammas.item(gram_matrix.list.var().arg_max())


def optimize_hyperparameters_by_cv(
    model,
    x: pl.DataFrame,
    y: pl.Series,
    y_true: pl.Series,
    cs: pl.Series,
    epsilons: pl.Series,
    gammas: pl.Series,
    cv: KFold,
    verbose: bool = True,
) -> tuple[float, float, float]:
    # ハイパーパラメータC, ε, γの最適化
    # グラム行列の分散が最大となるγを探索
    optimal_nonlinear_gamma = calc_optimal_gamma(x, gammas)

    # CV による ε の最適化
    calc_r2_params = {"model": model, "x": x, "y": y, "cv": cv, "y_true": y_true}
    # ε の最適化
    r2_cvs = pl.Series(
        "epsilons_r2",
        [
            calc_r2(
                **calc_r2_params,
                kernel="rbf",
                C=3,
                epsilon=epsilon,
                gamma=optimal_nonlinear_gamma,
            )
            for epsilon in epsilons
        ],
    )
    optimal_nonlinear_epsilon: float = epsilons.item(r2_cvs.arg_max())

    # CV による C の最適化
    r2_cvs = pl.Series(
        "cs_r2",
        [
            calc_r2(
                **calc_r2_params,
                kernel="rbf",
                C=c,
                epsilon=optimal_nonlinear_epsilon,
                gamma=optimal_nonlinear_gamma,
            )
            for c in cs
        ],
    )
    optimal_nonlinear_c: float = cs.item(r2_cvs.arg_max())
    # CV による γ の最適化
    r2_cvs = pl.Series(
        "gammas_r2",
        [
            calc_r2(
                **calc_r2_params,
                kernel="rbf",
                C=optimal_nonlinear_c,
                epsilon=optimal_nonlinear_epsilon,
                gamma=gamma,
            )
            for gamma in gammas
        ],
    )
    optimal_nonlinear_gamma: float = gammas.item(r2_cvs.arg_max())
    # 結果の確認
    if verbose:
        print(
            f"最適化された C : {optimal_nonlinear_c} (log(C)={math.log2(optimal_nonlinear_c)})"
        )
        print(
            f"最適化された ε : {optimal_nonlinear_epsilon} (log(ε)={math.log2(optimal_nonlinear_epsilon)})"
        )
        print(
            f"最適化された γ : {optimal_nonlinear_gamma} (log(γ)={math.log2(optimal_nonlinear_gamma)})"
        )
    return (
        optimal_nonlinear_c,
        optimal_nonlinear_epsilon,
        optimal_nonlinear_gamma,
    )


def optimize_hyperparameters_by_gs(
    model,
    x: pl.DataFrame,
    y: pl.Series,
    param_grid: dict,
    cv: KFold,
    **model_kwargs,
) -> dict:
    gs = GridSearchCV(estimator=model(**model_kwargs), param_grid=param_grid, cv=cv)
    gs.fit(x, y)
    for param_name, value in gs.best_params_.items():
        print(f"Best parameter log({param_name}): {math.log2(value)}")
    return gs.best_params_


def add_sqrt_and_interaction_terms(df: pl.DataFrame) -> pl.DataFrame:
    """説明変数の二乗項や交差項を追加"""
    from itertools import combinations

    return df.with_columns(  # 二乗項の追加
        [(pl.col(col) ** 2).alias(f"{col}^2") for col in df.columns]
    ).with_columns(  # 交差項の追加
        [
            (pl.col(col1) * pl.col(col2)).alias(f"{col1}*{col2}")
            for col1, col2 in combinations(df.columns, 2)
        ]
    )


def calc_ad_by_knn(
    x: pl.DataFrame,
    n_neighbors: int,
    ad_threshold: float = 0.96,
    remove_self: bool = True,
) -> tuple[pl.Series, pl.Series, float]:
    ad_model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    ad_model.fit(x)

    if remove_self:
        # 1(自分自身) + n_neighbors個のサンプルを抽出
        knn_distance_wt_self, _ = ad_model.kneighbors(x, n_neighbors=1 + n_neighbors)
        # 自分自身との距離を削除
        knn_distance = pl.DataFrame(knn_distance_wt_self).drop(pl.first())

    # 距離の平均を取得
    mean_of_knn_distance = knn_distance.mean_horizontal().rename("mean_of_knn_distance")
    # 各距離にたいしAD の中/外を判定
    ad_threshold = mean_of_knn_distance.quantile(ad_threshold, "lower")
    within_ad_flag = mean_of_knn_distance.le(ad_threshold)

    return mean_of_knn_distance, within_ad_flag, ad_threshold


def calc_ad_by_ocsvm(
    ad_model: OneClassSVM,
    x: pl.DataFrame,
    train_data: bool = True,
) -> tuple[pl.Series, pl.Series]:
    # トレーニングデータのデータ密度 (f(x) の値)を取得し、ADの中/外を判定(0以上:AD内、0未満:AD外)
    data_density = pl.Series(ad_model.decision_function(x))
    within_ad_flag = data_density.ge(0)

    number_of_outliers = data_density.lt(0).sum()
    if train_data:
        print(
            f"\nトレーニングデータにおけるサポートベクター数 :{ad_model.support_.size}"
            f"\nトレーニングデータにおけるサポートベクターの割合 :{ad_model.support_.size / x.height}"
            f"\nトレーニングデータにおける外れサンプル数 :{number_of_outliers}"
            f"\nトレーニングデータにおける外れサンプルの割合 :{number_of_outliers / x.height}"
        )
    else:
        print(
            f"\n予測用データセットにおける外れサンプル数 :{number_of_outliers}",
            f"\n予測用データセットにおける外れサンプルの割合 :{number_of_outliers / x.height}",
        )

    return data_density, within_ad_flag


def calc_acquisition_func(
    y: pl.Series,
    y_std: pl.Series,
    func: Literal["MI", "EI", "PI", "PTR"],
    target_range: tuple[float, float],
    relaxation: float,
    delta: float,
) -> pl.Series:
    # 比較用(temp)

    # MI で必要な "ばらつき" を 0 で初期化
    cumulative_variance = pl.zeros(y.len(), eager=True)

    if func == "MI":
        ac_func_pred = y + np.log(2 / delta) ** 0.5 * (
            (y**2 + cumulative_variance) ** 0.5 - cumulative_variance**0.5
        )
        cumulative_variance = cumulative_variance + y_std**2
    elif func == "EI":
        ac_func_pred = (y - max(y) - relaxation * y.std()) * norm.cdf(
            (y - max(y) - relaxation * y.std()) / y_std
        ) + y_std * norm.pdf((y - max(y) - relaxation * y.std()) / y_std)
    elif func == "PI":
        ac_func_pred = norm.cdf((y - max(y) - relaxation * y.std()) / y_std)
    elif func == "PTR":
        ac_func_pred = norm.cdf(target_range[1], loc=y, scale=y_std) - norm.cdf(
            target_range[0], loc=y, scale=y_std
        )

    ac_func_pred[y_std <= 0] = 0
    ac_func_pred = (
        pl.Series(ac_func_pred)
        if not isinstance(ac_func_pred, pl.Series)
        else ac_func_pred
    )
    return ac_func_pred, cumulative_variance
