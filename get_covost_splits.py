# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import csv
from pathlib import Path

import pandas as pd


XX_EN_LANGUAGES = {
        "1": ["fr", "de", "nl", "ru", "es", "it", "tr", "fa", "sv-SE", "mn",
              "zh-CN"],
        "2": ["fr", "de", "es", "ca", "it", "ru", "zh-CN", "pt", "fa", "et",
              "mn", "nl", "tr", "ar", "sv-SE", "lv", "sl", "ta", "ja", "id",
              "cy"],
        "3": ["ar", "as", "az", "be", "bg", "bn", "br", "ca", "ckb", "cs", "cy", 
              "da", "de", "dv", "el", "es", "et", "eu", "fa", "fi", "fr", "fy-NL", 
              "ga-IE", "ha", "hi", "hu", "hy-AM", "id", "ig", "it", "ja", "ka", "kab", 
              "kk", "ky", "lg", "lt", "lv", "mk", "ml", "mn", "mr", "mt", "ne-NP", 
              "nl", "or", "pa-IN", "pl", "pt", "ro", "ru", "sk",  "sl", "sr", "sv-SE", 
              "sw", "ta", "th", "tr", "tt", "uk", "ur", "uz", "vi", "yue", "zh-CN", "zh-HK", 
              "zh-TW"]
}
EN_XX_LANGUAGES = {
        "1": [],
        "2": ["de", "tr", "fa", "sv-SE", "mn", "zh-CN", "cy", "ca", "sl", "et",
              "id", "ar", "ta", "lv", "ja"],
        "3": ["ar", "as", "az", "be", "bg", "bn", "ca", "ckb", "cs", "cy", "da", 
              "de", "el", "en-uk", "es", "et", "fa", "fi", "fr", "ha", "hi", "hu", 
              "hy-AM", "id", "ig", "it", "ja", "ka", "kab", "kk", "ky", "lg", "lt", 
              "lv", "mk", "ml", "mn", "mr", "mt", "ne-NP", "nl", "or", "pa-IN", "pl", 
              "pt-BR", "ro", "ru", "rw", "sc", "sk", "sl", "sr", "sv-SE", "sw", "ta", 
              "th", "ti", "tr", "tt", "tw", "ur", "uz", "vi", "yue", "zh-CN", "zh-TW"],
}

XX_XX_DIRECTIONS= {
    "3" : ["de-tr", "hi-ta", "hu-ro", "it-sl", "ja-ru", "ka-uk", "lt-sv", "lv-ru", "mn-zh",
           "mt-sl", "ro-hu", "ru-ja", "ru-lv", "sl-it", "sl-mt"],
}

SPLITS = ["train", "dev", "test"]
TSV_PREFIX = {"1": "covost", "2": "covost_v2", "3": "covost_v3"}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", "-v", type=str, choices=["1", "2"],
                        required=True, help="CoVoST version")
    parser.add_argument("--src-lang", "-s", type=str, required=True,
                        help="source language code")
    parser.add_argument("--tgt-lang", "-t", type=str, required=True,
                        help="target language code")
    parser.add_argument("--root", "-d", type=str, required=True,
                        help="root path to translation TSV and output TSVs")
    parser.add_argument("--cv-tsv", type=str, required=True,
                        help="path to validated.tsv from Common Voice")
    return parser.parse_args()


def load_df_from_tsv(path: Path):
    return pd.read_csv(path, sep="\t", header=0, encoding="utf-8",
                       escapechar="\\", quoting=csv.QUOTE_NONE, na_filter=False)


def save_df_to_tsv(dataframe, path: Path):
    dataframe.to_csv(path, sep="\t", header=True, index=False, encoding="utf-8",
                     escapechar="\\", quoting=csv.QUOTE_NONE)


def get_v1_split(df: pd.DataFrame, split: str):
    return df[(df["split"] == split) | (df["split"] == f"{split}_covost")]


def get_v2_split(df: pd.DataFrame, split: str):
    if split == "train":
        return df[(df["split"] == split) | (df["split"] == f"{split}_covost")]
    else:
        return df[df["split"] == split]

def get_v3_split(df: pd.DataFrame, split: str):
    return df[(df["split"] == split) | (df["split"] == f"{split}_covost")]

def main():
    args = get_args()

    ver, src, tgt = args.version, args.src_lang, args.tgt_lang
    assert src != tgt
    if ver < 3: 
        assert "en" in {src, tgt}
        if src == "en":
            assert tgt in EN_XX_LANGUAGES[ver]
        else:
            assert src in XX_EN_LANGUAGES[ver]
    else: 
        if "en" in {src, tgt}: 
            if src == "en":
                assert tgt in EN_XX_LANGUAGES[ver]
            else:
                assert src in XX_EN_LANGUAGES[ver]
        else: 
            assert src + "-" + tgt in XX_XX_DIRECTIONS[ver]
            

    root = Path(args.root)
    tsv_prefix = TSV_PREFIX[ver]
    cv_tsv = load_df_from_tsv(args.cv_tsv)
    covost_tsv = load_df_from_tsv(root / f"{tsv_prefix}.{src}_{tgt}.tsv")

    df = pd.merge(
        left=cv_tsv[["path", "sentence", "client_id"]],
        right=covost_tsv[["path", "translation", "split"]],
        how="inner",
        on="path",
    )
    for split in SPLITS:
        get_split = {"1": get_v1_split, "2": get_v2_split, "3": get_v3_split}.get(ver, None)
        assert get_split is not None
        cur_df = get_split(df, split)
        cur_df = cur_df[["path", "sentence", "translation", "client_id"]]
        save_df_to_tsv(cur_df, root / f"{tsv_prefix}.{src}_{tgt}.{split}.tsv")


if __name__ == "__main__":
    main()
