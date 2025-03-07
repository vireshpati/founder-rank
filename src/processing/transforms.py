import numpy as np
import pandas as pd
from src.utils.matrix_utils import get_tier


class ProfileTransforms:
    def __init__(self, data, matrix):
        self.data = data
        self.MATRIX = matrix
        self.df = None

    def _extract_undergrad_school(self, edu_list, cutoff_date=None):
        for ed in edu_list:
            if (
                cutoff_date
                and ed.get("ends_at")
                and ed.get("ends_at").get("year")
                and int(ed.get("ends_at").get("year")) > cutoff_date
            ):
                continue

            deg = (ed.get("degree_name") or "").lower()
            fos = (ed.get("field_of_study") or "").lower()
            if any(
                (keyword in deg or keyword in fos)
                for keyword in [
                    "bs",
                    "ba",
                    "bachelor",
                    "bse",
                    "bsba",
                ]
            ):
                return ed.get("school")
        return None

    def _extract_grad_school(self, edu_list, cutoff_date=None):
        for ed in edu_list:
            if (
                cutoff_date
                and ed.get("ends_at")
                and ed.get("ends_at").get("year")
                and int(ed.get("ends_at").get("year")) > cutoff_date
            ):
                continue

            fos = (ed.get("field_of_study") or "").lower()
            deg = (ed.get("degree_name") or "").lower()
            if any(
                (keyword in deg or keyword in fos)
                for keyword in [
                    "master",
                    "mba",
                    "ms",
                    "phd",
                ]
            ):
                return ed.get("school")
        return None

    def _extract_current_experience(self, experiences, cutoff_date=None):
        if cutoff_date:
            for exp in experiences:
                start_year = exp.get("starts_at", {}).get("year")
                end_year = exp.get("ends_at", {}).get("year") if exp.get("ends_at") else None

                if start_year and int(start_year) <= cutoff_date and (not end_year or int(end_year) >= cutoff_date):
                    return (exp.get("company"), exp.get("title"))
            return (None, None)
        else:
            current_exp = next((exp for exp in experiences if exp.get("ends_at") is None), None)
            if current_exp:
                return (current_exp.get("company"), current_exp.get("title"))
            return (None, None)

    def _extract_previous_experience(self, experiences, cutoff_date=None):
        if cutoff_date:
            previous_exps = []
            for exp in experiences:
                start_year = exp.get("starts_at", {}).get("year")
                end_year = exp.get("ends_at", {}).get("year") if exp.get("ends_at") else None

                if start_year and end_year and int(end_year) <= cutoff_date:
                    previous_exps.append((exp.get("company"), exp.get("title")))

            return previous_exps if previous_exps else []
        else:
            if len(experiences) > 1:
                return [(exp.get("company"), exp.get("title")) for exp in experiences[1:]]
            return []

    def _process_profiles(self, cutoff_date=None):
        records = []
        for result in self.data.get("results", []):  # Use get with default
            try:
                profile = result.get("profile", {})
                full_name = profile.get("full_name")
                edu_list = profile.get("education", [])
                exp_list = profile.get("experiences", [])
                linkedin = result.get("linkedin_profile_url", [])

                undergrad = self._extract_undergrad_school(edu_list, cutoff_date)
                grad = self._extract_grad_school(edu_list, cutoff_date)
                current_company, current_title = self._extract_current_experience(exp_list, cutoff_date)
                previous_experiences_titles = self._extract_previous_experience(exp_list, cutoff_date)

                row = {
                    "Name": full_name,
                    "Undergrad School": undergrad,
                    "Graduate School": grad,
                    "Current Company": current_company,
                    "Current Title": current_title,
                    "Previous Companies": (
                        [exp[0] for exp in previous_experiences_titles]
                        if isinstance(previous_experiences_titles, list)
                        else []
                    ),
                    "Previous Titles": (
                        [exp[1] for exp in previous_experiences_titles]
                        if isinstance(previous_experiences_titles, list)
                        else []
                    ),
                    "Linkedin": linkedin,
                }
                records.append(row)
            except Exception as e:
                print(f"Error processing profile: {e}")
                continue

        self.df = pd.DataFrame(records)

    def _add_ordinal_columns(self):
        # Undergrad
        self.df["UNDERGRAD"] = np.where(
            self.df["Undergrad School"].isin(
                get_tier(
                    self.MATRIX,
                    "UNDERGRAD",
                    3,
                )
            ),
            3,
            np.where(
                self.df["Undergrad School"].isin(
                    get_tier(
                        self.MATRIX,
                        "UNDERGRAD",
                        2,
                    )
                ),
                2,
                1,
            ),
        )

        # Graduate
        self.df["GRADUATE"] = np.where(
            self.df["Graduate School"].isin(
                get_tier(
                    self.MATRIX,
                    "GRADUATE",
                    3,
                )
            ),
            3,
            np.where(
                self.df["Graduate School"].isin(
                    get_tier(
                        self.MATRIX,
                        "GRADUATE",
                        2,
                    )
                ),
                2,
                np.where(
                    self.df["Graduate School"].notnull() & (self.df["Graduate School"] != "None"),
                    1,
                    0,
                ),
            ),
        )

        # Company Quality
        self.df["COMPANY"] = self.df.apply(
            lambda row: (
                3
                if any(
                    company
                    in get_tier(
                        self.MATRIX,
                        "COMPANY",
                        3,
                    )
                    for company in row["Previous Companies"]
                    if company
                )
                else (
                    2
                    if any(
                        company
                        in get_tier(
                            self.MATRIX,
                            "COMPANY",
                            2,
                        )
                        for company in row["Previous Companies"]
                        if company
                    )
                    else 1
                )
            ),
            axis=1,
        )

        # Seniority
        self.df["SENIORITY"] = self.df.apply(
            lambda row: (
                3
                if any(
                    title
                    and any(
                        keyword.lower() in title.lower()
                        for keyword in get_tier(
                            self.MATRIX,
                            "SENIORITY",
                            3,
                        )
                    )
                    for title in row["Previous Titles"]
                    if title
                )
                else (
                    2
                    if any(
                        title
                        and any(
                            keyword.lower() in title.lower()
                            for keyword in get_tier(
                                self.MATRIX,
                                "SENIORITY",
                                2,
                            )
                        )
                        for title in row["Previous Titles"]
                        if title
                    )
                    else 1
                )
            ),
            axis=1,
        )

        # Expertise
        self.df["EXPERTISE"] = self.df.apply(
            lambda row: (
                3
                if any(
                    title
                    and any(
                        kw.lower() in word.lower()
                        for word in title.split()
                        for kw in get_tier(
                            self.MATRIX,
                            "EXPERTISE",
                            3,
                        )
                    )
                    for title in row["Previous Titles"]
                    if title
                )
                else (
                    2
                    if any(
                        title
                        and any(
                            kw.lower() in word.lower()
                            for word in title.split()
                            for kw in get_tier(
                                self.MATRIX,
                                "EXPERTISE",
                                2,
                            )
                        )
                        for title in row["Previous Titles"]
                        if title
                    )
                    else 1
                )
            ),
            axis=1,
        )

    def _add_ai_evaluations(self, perplexity_client):
        ai_evaluations = self.df.apply(lambda row: perplexity_client.eval_person(row, self.MATRIX), axis=1)
        self.df["EXIT"] = ai_evaluations.apply(lambda x: x.get("exited_founder", 0))
        self.df["FOUNDER"] = ai_evaluations.apply(lambda x: x.get("previous_founder", 1))
        self.df["STARTUP"] = ai_evaluations.apply(lambda x: x.get("startup_experience", 1))

    def transform(self, perplexity_client=None, cutoff_date=None):
        self._process_profiles(cutoff_date)
        self._add_ordinal_columns()
        if perplexity_client:
            self._add_ai_evaluations(perplexity_client)
        return self.df


    def create_feature_matrix(self):
        one_hot_matrices = []
        for cat, cfg in self.MATRIX.items():
            dim = cfg["DIMENSION"]
            values = self.df[cat].to_numpy()  # Ordinal values in {0,1,2,3}
            matrix = one_hot_encode_column(values, dim)
            one_hot_matrices.append(matrix)

        feature_matrix = np.concatenate(one_hot_matrices, axis=1)
        return feature_matrix

def one_hot_encode_column(values, dimension):
        indices = values.copy()
        if dimension == 3:
            indices = values - 1
        
        indices = np.clip(indices, 0, dimension - 1)
        return np.eye(dimension, dtype=int)[indices.astype(int)]
