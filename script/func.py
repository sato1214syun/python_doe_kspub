import math
from pathlib import Path

import numpy as np
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
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


# カーネル 11 種類
def generate_kernels(x: np.ndarray) -> list[Sum]:
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


def load_data(
    file_path: Path | str, index: int | str | None = None
) -> tuple[pl.DataFrame, pl.Series]:
    """Load data from a CSV file."""
    dataset = pl.read_csv(file_path)
    if index is None:
        return dataset, None
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


def autoscaling[T, V](data: T, data2: V = None) -> T:
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


def rescaling[T, V](autoscaled_data: T, original_data: V, is_std: bool = False) -> T:
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


def scatter_plot_of_result[T: pl.DataFrame | pl.Series](
    save_path: Path | str, x: T, y: T, x_label: str = "x", y_label: str = "y"
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
    x: pl.DataFrame,
    y: pl.Series,
    model,
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
    calc_r2_params = {"x": x, "y": y, "model": model, "cv": cv, "y_true": y_true}
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
