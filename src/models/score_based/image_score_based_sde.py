import os

from mnist_classifier import get_module
from src.data import Dataset
from src.models.image_model import ImageModel
from src.report import *
from src.util import get_available_device

from .score_based_sde import ScoreBasedSde


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    min_values = images.view(images.shape[0], -1).min(dim=1)[0].view(-1, 1, 1, 1)
    max_values = images.view(images.shape[0], -1).max(dim=1)[0].view(-1, 1, 1, 1)
    return (images - min_values) / (max_values - min_values)


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
        samples = normalize_images(samples.unsqueeze(1)).squeeze()
        images_folder = str(path) + "/samples"
        os.makedirs(images_folder)

        store_images(samples, folder=images_folder)

        plt.imshow(samples[0])
        plt.colorbar()
        plt.savefig(f"image.{IMAGE_FORMAT}")
        plt.close()
