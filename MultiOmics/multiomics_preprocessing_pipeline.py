
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class MultiOmicsPreprocessor:
    def __init__(self, protein_min_coverage=0.70):
        self.protein_min_coverage = protein_min_coverage
        self.models = {}

    def fit(self, matrices, train_ids):
        self.models = {}

        for name, df in matrices.items():
            train = df.loc[train_ids].apply(
                pd.to_numeric, errors="coerce"
            )

            if name == "proteomics":
                coverage = train.notna().mean()
                selected = coverage[
                    coverage >= self.protein_min_coverage
                ].index
            else:
                selected = train.columns[
                    train.notna().any(axis=0)
                ]

            train = train[selected]

            imputer = SimpleImputer(
                strategy="median",
                keep_empty_features=True
            )
            filled = imputer.fit_transform(train)

            variance = filled.var(axis=0)
            keep = variance > 0

            selected = selected[keep]
            filled = filled[:, keep]

            if len(selected) == 0:
                raise ValueError(
                    f"No features retained for {name}"
                )

            scaler = None

            if name in ["expression", "proteomics"]:
                scaler = StandardScaler()
                scaler.fit(filled)

            self.models[name] = {
                "features": selected.tolist(),
                "imputer": imputer,
                "keep": keep,
                "scaler": scaler
            }

        return self

    def transform(self, matrices, sample_ids):
        output = {}

        for name, model in self.models.items():
            df = matrices[name].loc[sample_ids]

            original_features = (
                model["imputer"].feature_names_in_
            )

            data = df.reindex(
                columns=original_features
            ).apply(pd.to_numeric, errors="coerce")

            values = model["imputer"].transform(data)
            values = values[:, model["keep"]]

            if model["scaler"] is not None:
                values = model["scaler"].transform(values)

            output[name] = pd.DataFrame(
                values,
                index=sample_ids,
                columns=model["features"]
            )

        return output

    def fit_transform(self, matrices, train_ids):
        self.fit(matrices, train_ids)
        return self.transform(matrices, train_ids)
