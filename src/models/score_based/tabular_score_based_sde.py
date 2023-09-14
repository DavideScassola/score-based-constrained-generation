import os

from src.data import Dataset
from src.models.tabular_model import TabularModel
from src.preprocessors.to_tensor import ToTensor
from src.report import *
from src.util import rejection_sampling, store_json

from .score_based_sde import ScoreBasedSde


def constraint_rejection_sampling(df: pd.DataFrame, constraint: Constraint):
    probability_distribution = lambda x: np.exp(
        constraint.strength * constraint.f(torch.from_numpy(x)).numpy()
    )
    x = rejection_sampling(df.to_numpy(), probability_distribution)
    return pd.DataFrame(x, columns=df.columns)


def refill_rejection_sampling(
    df: pd.DataFrame, constraint: Constraint, sampler, generation_options
):
    df_rejection_sampling = constraint_rejection_sampling(df, constraint)

    discarded_samples = len(df) - len(df_rejection_sampling)
    # TODO: make this batched
    if discarded_samples > 0:
        acceptance_rate = len(df_rejection_sampling) / len(df)

        generation_options["n_samples"] = int(
            1.2 * discarded_samples // acceptance_rate
        )

        print("not enough samples after rejection, generating again ...")
        df_new = sampler(**generation_options)
        df_rejection_sampling_new = constraint_rejection_sampling(df_new, constraint)
        df_rejection_sampling = pd.concat(
            [df_rejection_sampling, df_rejection_sampling_new]
        )

    return df_rejection_sampling.iloc[: len(df)], len(df_rejection_sampling) / (
        len(df_new) + len(df)
    )


SHOW_REJECTION_SAMPLING = True


class TabularScoreBasedSde(ScoreBasedSde, TabularModel):
    def generate_report(
        self,
        *,
        path: str | Path,
        dataset: Dataset,
        generation_options: dict,
        constraint: Constraint | None = None
    ):
        report_folder = path / Path(REPORT_FOLDER_NAME)
        os.makedirs(report_folder, exist_ok=False)

        samples = self.generate(**generation_options)
        store_samples(df_generated=samples, path=report_folder, name="guidance.csv")

        if constraint:
            # TODO: setting the score to None temporarily is not so elegant
            original_score = self.score_model.constraint
            self.score_model.constraint = None
            df_train = self.generate(**generation_options)
            store_samples(
                df_generated=df_train, path=report_folder, name="no_guidance.csv"
            )

            if SHOW_REJECTION_SAMPLING:
                df_train, acceptance_rate = refill_rejection_sampling(
                    df_train, constraint, self.generate, generation_options
                )
                store_json(
                    {"acceptance_rate": acceptance_rate},
                    file=report_folder / Path("acceptance_rate.json"),
                )
                self.score_model.constraint = original_score
                store_samples(
                    df_generated=df_train,
                    path=report_folder,
                    name="rejection_sampling.csv",
                )

            if len(samples.columns) > 1:
                # TODO: exp?
                constraint_exp = lambda x: np.exp(
                    constraint.strength
                    * constraint.f(ToTensor().fit_transform(x)).numpy()
                )
                df_train["constraint_value"] = constraint_exp(df_train)
                samples["constraint_value"] = constraint_exp(samples)
                rcParams.update({"figure.autolayout": True})
        else:
            df_train = dataset.get(train=True)

        print("plotting...")
        for comparison in (
            histograms_comparison,
            statistics_comparison,
            correlations_comparison,
        ):
            comparison(
                df_generated=samples,
                name_generated="guidance" if constraint else "generated",
                df_train=df_train,
                name_original=(
                    "rejection sampling" if SHOW_REJECTION_SAMPLING else "no guidance"
                )
                if constraint
                else "true",
                path=report_folder,
                constraint=None if SHOW_REJECTION_SAMPLING else constraint,
                model=self,
            )

        self.specific_report_plots(report_folder)
