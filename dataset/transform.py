import numpy as np
from monai import transforms as mt
from monai.transforms import MapTransform


class ReorganizeTransform(MapTransform):
    def __init__(self, required_keys=("T1", "T1CE", "T2", "FLAIR", "seg")):
        """
        Partitions baseline and followup images.
        :param required_keys: image keys to partition
        """
        super().__init__(keys=None)
        self.required_keys = required_keys

    def __call__(self, data):
        if isinstance(data, list):
            data = data[0]
        baseline, followup = {}, {}
        for key in self.required_keys:
            baseline_key = f"baseline_{key}"
            followup_key = f"followup_{key}"
            assert baseline_key in data
            assert followup_key in data
            baseline[key] = data[baseline_key]
            followup[key] = data[followup_key]
            del data[baseline_key]
            del data[followup_key]

        data.update({"baseline": baseline, "followup": followup})

        return data


class CropAround3DMaskd(mt.MapTransform):
    def __init__(self, keys, mask_key, margin=10):
        super().__init__(keys)
        self.mask_key = mask_key
        self.margin = margin

    def __call__(self, data):
        d = dict(data)
        mask = d[self.mask_key][0]  # assuming mask shape (C=1, H, W, D)
        # Find bounding box of non-zero voxels
        nonzero = np.nonzero(mask)
        minz, maxz = nonzero[0].min(), nonzero[0].max()
        miny, maxy = nonzero[1].min(), nonzero[1].max()
        minx, maxx = nonzero[2].min(), nonzero[2].max()

        # Expand bounding box by margin, ensuring within image bounds
        shape = mask.shape
        minz = max(minz - self.margin, 0)
        maxz = min(maxz + self.margin + 1, shape[0])
        miny = max(miny - self.margin, 0)
        maxy = min(maxy + self.margin + 1, shape[1])
        minx = max(minx - self.margin, 0)
        maxx = min(maxx + self.margin + 1, shape[2])

        # Crop all keys accordingly
        for key in self.keys:
            img = d[key]
            # img shape assumed (C, H, W, D)
            d[key] = img[:, minz:maxz, miny:maxy, minx:maxx]

        return d
