import os
import sys

from natsort import natsorted
from PIL import Image

TRANSPOSE = True


def make_grid(image_folder, rows, cols):
    images = []
    for filename in natsorted(os.listdir(image_folder)):
        if filename.endswith(".png"):
            img_path = os.path.join(image_folder, filename)
            img = Image.open(img_path)
            images.append(img)

    if len(images) < rows * cols:
        print("Not enough images in the folder.")
        return

    collage_width = images[0].width * cols
    collage_height = images[0].height * rows
    collage = Image.new("RGB", (collage_width, collage_height))

    for row in range(rows):
        for col in range(cols):
            index = row * cols + col if not TRANSPOSE else col * rows + row
            collage.paste(
                images[index], (col * images[0].width, row * images[0].height)
            )

    collage.save(f"{image_folder}/../collage_{rows}x{cols}.png")
    print("Collage saved as collage.png")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python make_grid.py path/to/image_folder <rows> <cols>")
    else:
        image_folder = sys.argv[1]
        rows = int(sys.argv[2])
        cols = int(sys.argv[3])
        make_grid(image_folder, rows, cols)
