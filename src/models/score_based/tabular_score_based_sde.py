import os

from src.data import Dataset
from src.models.tabular_model import TabularModel
from src.preprocessors.preprocessor import Preprocessor
from src.preprocessors.to_tensor import ToTensor
from src.report import *
from src.util import edit_json, rejection_sampling, store_json

from .score_based_sde import ScoreBasedSde


def constraint_rejection_sampling(
    df: pd.DataFrame, constraint: Constraint, to_tensor_preprocessor: Preprocessor
):
    if constraint.hard_f is None:
        probability_distribution = lambda x: np.exp(
            constraint.strength * constraint.f(torch.from_numpy(x)).numpy()
        )
        x_tensor = to_tensor_preprocessor.transform(df)
        x_numpy = x_tensor.cpu().numpy()
        x = rejection_sampling(x_numpy, probability_distribution)
        x_tensor = torch.from_numpy(x)
        return to_tensor_preprocessor.reverse_transform(x_tensor)  # type: ignore
    else:
        return df[constraint.hard_f(df)].copy()


def refill_rejection_sampling(
    df: pd.DataFrame,
    constraint: Constraint,
    sampler,
    generation_options,
    to_tensor_preprocessor,
):
    df_rejection_sampling = constraint_rejection_sampling(
        df, constraint, to_tensor_preprocessor
    )

    discarded_samples = len(df) - len(df_rejection_sampling)
    # TODO: make this batched
    if discarded_samples > 0:
        acceptance_rate = len(df_rejection_sampling) / len(df)

        generation_options["n_samples"] = int(
            1.2 * discarded_samples // acceptance_rate
        )

        print("not enough samples after rejection, generating again ...")
        df_new = sampler(**generation_options)
        df_rejection_sampling_new = constraint_rejection_sampling(
            df_new, constraint, to_tensor_preprocessor
        )
        df_rejection_sampling = pd.concat(
            [df_rejection_sampling, df_rejection_sampling_new]
        )
    else:
        return df_rejection_sampling.iloc[: len(df)], 1.0

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
        name_generated = "guidance.csv" if constraint else "generated.csv"

        samples = self.generate(**generation_options)
        store_samples(df_generated=samples, path=report_folder, name=name_generated)

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
                    df_train,
                    constraint,
                    self.generate,
                    generation_options,
                    self.df2tensor,
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
                    * constraint.f(self.df2tensor.transform(x)).numpy()
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

        if len(samples.shape) == 2 and samples.shape[1] == 2:
            coordinates_comparison(
                df_generated=samples,
                df_train=df_train,
                path=report_folder,
                model=self,
            )

        self.specific_report_plots(report_folder)

        if samples.shape[1] > 1:
            summary_report(path=report_folder)

        if constraint is not None and constraint.hard_f is not None:
            with edit_json(report_folder / "stats_summary.json") as stats_summary:
                stats_summary["guidance_sat"] = constraint.hard_f(samples).mean()
                stats_summary["rejection_sampling_sat"] = constraint.hard_f(
                    df_train
                ).mean()
