import numpy as np
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    Matern,
    WhiteKernel,
)
import polars as pl
from pathlib import Path


# カーネル 11 種類
def generate_kernels(x: np.ndarray) -> list:
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
    file_path: Path | str, y_col: int | str | None = None
) -> tuple[pl.DataFrame, pl.Series]:
    """Load data from a CSV file."""
    dataset = pl.read_csv(file_path)
    if y_col is None:
        return dataset, None
    if isinstance(y_col, int):
        y_col = dataset.columns[y_col]
    y = dataset.get_column(y_col)
    x = dataset.drop(y.name)
    return x, y


def save_csv(
    data: pl.DataFrame,
    file_path: Path | str,
    with_index: bool = False,
    index_name: str = "",
    offset: int = 0,
) -> None:
    """Save DataFrame to a CSV file."""
    if with_index:
        data = data.clone().with_row_index(name=index_name, offset=offset)
    data.write_csv(file_path)
