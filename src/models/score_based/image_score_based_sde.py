import os

from src.data import Dataset
from src.models.image_model import ImageModel
from src.report import *

from .score_based_sde import ScoreBasedSde


class ImageScoreBasedSde(ScoreBasedSde, ImageModel):
    def generate_report(
        self,
        *,
        path: str | Path,
        dataset: Dataset,
        generation_options: dict,
        constraint: Constraint | None = None,
    ):
        report_folder = path / Path(REPORT_FOLDER_NAME)
        os.makedirs(report_folder, exist_ok=False)

        samples = self.generate(**generation_options).cpu()
        images_folder = str(path) + "/samples"
        os.makedirs(images_folder)

        store_images(samples, folder=images_folder)

        # if constraint:
        #    print(np.exp(constraint.f(samples).numpy()))
        plt.imshow(samples[0])
        plt.colorbar()
        plt.savefig(f"image.{IMAGE_FORMAT}")
        plt.close()
