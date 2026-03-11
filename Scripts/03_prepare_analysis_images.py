import pandas as pd

INPUT = r"D:\photo_ai\data\index\unique_media.csv"
OUT = r"D:\photo_ai\data\index\analysis_images.csv"


def main():
    df = pd.read_csv(INPUT)

    imgs = df[
        (df["is_image"] == True) &
        (df["phash"].notna())
    ].copy()

    imgs.to_csv(OUT, index=False, encoding="utf-8-sig")

    print("rows =", len(df))
    print("images_for_analysis =", len(imgs))
    print("saved_to =", OUT)


if __name__ == "__main__":
    main()