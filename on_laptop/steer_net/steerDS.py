import csv
from os import path
from glob import glob

import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class SteerDataSet(Dataset):
    def __init__(
        self,
        root_folder,
        img_ext=".jpg",
        transform=None,
        manifest_csv=None,
        image_col="image_path",
        steering_col="steering",
    ):
        self.root_folder = root_folder
        self.transform = transform
        self.img_ext = img_ext
        self.totensor = transforms.ToTensor()

        self.samples = []
        if manifest_csv is not None:
            with open(manifest_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rel_or_abs = row[image_col]
                    image_path = rel_or_abs if path.isabs(rel_or_abs) else path.join(self.root_folder, rel_or_abs)
                    self.samples.append(
                        {
                            "image_path": image_path,
                            "steering": np.float32(row[steering_col]),
                            "frame_id": row.get("frame_id"),
                        }
                    )
        else:
            filenames = glob(path.join(self.root_folder, "*" + self.img_ext))
            filenames.sort()
            for file_name in filenames:
                steering = file_name.split("/")[-1].split(self.img_ext)[0][6:]
                self.samples.append(
                    {
                        "image_path": file_name,
                        "steering": np.float32(steering),
                        "frame_id": None,
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = cv2.imread(sample["image_path"])

        if img is None:
            raise RuntimeError(f"Could not load image: {sample['image_path']}")

        if self.transform is None:
            img = self.totensor(img)
        else:
            img = self.transform(img)

        return {
            "image": img,
            "steering": sample["steering"],
            "frame_id": sample["frame_id"],
        }


def test():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    ds = SteerDataSet("../dev_data/training_data", ".jpg", transform)

    print("The dataset contains %d images " % len(ds))

    ds_dataloader = DataLoader(ds, batch_size=1, shuffle=True)
    for sample in ds_dataloader:
        print(sample["image"].shape)
        print(sample["steering"])
        break


if __name__ == "__main__":
    test()
