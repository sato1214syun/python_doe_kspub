import polars as pl

def main():
    print("Hello from python-doe-kspub!")
    df = pl.read_csv("test_data/resin.csv")
    print(df.head())


if __name__ == "__main__":
    main()
