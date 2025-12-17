#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "awkward",
#     "numpy",
#     "polars",
#     "uproot",
# ]
# ///
import awkward as ak
import numpy as np
import polars as pl
import uproot


def read_parquet() -> pl.DataFrame:
    return pl.read_parquet('data_f32.parquet')


def convert_to_f64(df: pl.DataFrame) -> pl.DataFrame:
    return df.lazy().select(*[pl.col(col).cast(pl.Float64) for col in df.columns]).collect()


def write_root(tree: str = 'events'):
    df_parquet_f32 = read_parquet()
    file_root_f32 = uproot.recreate('data_f32.root')
    file_root_f32.mktree(tree, df_parquet_f32.to_dict())
    df_parquet_f64 = convert_to_f64(df_parquet_f32)
    file_root_f64 = uproot.recreate('data_f64.root')
    file_root_f64.mktree(tree, df_parquet_f64.to_dict())


def write_amptools(tree: str = 'kin'):
    df_parquet_f32 = read_parquet()
    file_amptools = uproot.recreate('data_amptools.root')
    df_dict_f32 = df_parquet_f32.to_dict()
    df_amptools_dict = {
        'E_Beam': df_dict_f32['beam_e'],
        'Px_Beam': df_dict_f32['beam_px'],
        'Py_Beam': df_dict_f32['beam_py'],
        'Pz_Beam': df_dict_f32['beam_pz'],
        'NumFinalState': np.full(len(df_dict_f32['proton_e']), 3, dtype=np.int32),
        'E_FinalState': ak.Array(
            np.array([df_dict_f32['proton_e'], df_dict_f32['kshort1_e'], df_dict_f32['kshort2_e']]).transpose().tolist()
        ),
        'Px_FinalState': ak.Array(
            np.array([df_dict_f32['proton_px'], df_dict_f32['kshort1_px'], df_dict_f32['kshort2_px']])
            .transpose()
            .tolist()
        ),
        'Py_FinalState': ak.Array(
            np.array([df_dict_f32['proton_py'], df_dict_f32['kshort1_py'], df_dict_f32['kshort2_py']])
            .transpose()
            .tolist()
        ),
        'Pz_FinalState': ak.Array(
            np.array([df_dict_f32['proton_pz'], df_dict_f32['kshort1_pz'], df_dict_f32['kshort2_pz']])
            .transpose()
            .tolist()
        ),
        'Weight': df_dict_f32['weight'],
    }
    file_amptools.mktree(tree, df_amptools_dict)
    file_amptools_pol = uproot.recreate('data_amptools_pol.root')
    df_amptools_pol_dict = {
        'E_Beam': df_dict_f32['beam_e'],
        'Px_Beam': df_dict_f32['pol_magnitude'] * np.cos(df_dict_f32['pol_angle']),
        'Py_Beam': df_dict_f32['pol_magnitude'] * np.sin(df_dict_f32['pol_angle']),
        'Pz_Beam': df_dict_f32['beam_pz'],
        'NumFinalState': np.full(len(df_dict_f32['proton_e']), 3, dtype=np.int32),
        'E_FinalState': ak.Array(
            np.array([df_dict_f32['proton_e'], df_dict_f32['kshort1_e'], df_dict_f32['kshort2_e']]).transpose().tolist()
        ),
        'Px_FinalState': ak.Array(
            np.array([df_dict_f32['proton_px'], df_dict_f32['kshort1_px'], df_dict_f32['kshort2_px']])
            .transpose()
            .tolist()
        ),
        'Py_FinalState': ak.Array(
            np.array([df_dict_f32['proton_py'], df_dict_f32['kshort1_py'], df_dict_f32['kshort2_py']])
            .transpose()
            .tolist()
        ),
        'Pz_FinalState': ak.Array(
            np.array([df_dict_f32['proton_pz'], df_dict_f32['kshort1_pz'], df_dict_f32['kshort2_pz']])
            .transpose()
            .tolist()
        ),
        'Weight': df_dict_f32['weight'],
    }
    file_amptools_pol.mktree(tree, df_amptools_pol_dict)


def main():
    df_parquet_f32 = read_parquet()
    convert_to_f64(df_parquet_f32).write_parquet('data_f64.parquet')
    write_root()
    write_amptools()


if __name__ == '__main__':
    main()
