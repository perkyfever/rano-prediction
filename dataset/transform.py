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
        baseline, followup = {}, {}
        for key in self.required_keys:
            baseline_key = f"baseline_{key}"
            followup_key = f"followup_{key}"
            assert baseline_key in data
            assert followup_key in data
            baseline[key] = data[baseline_key].permute(1, 0, 2)  # transform to 218x182x182
            followup[key] = data[followup_key].permute(1, 0, 2)  # transform to 218x182x182
            del data[baseline_key]
            del data[followup_key]

        data.update({"baseline": baseline, "followup": followup})

        return data
