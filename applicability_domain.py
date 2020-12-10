# Author: Md Mahmudulla Hassan
# CS@UTEP and SOP@UTEP
# Last modified: 12/09/2020

# Reference: "On a simple approach for determining applicability domain of QSAR models"
# DOI:10.1016/J.CHEMOLAB.2015.04.013
import numpy as np


class ApplicabilityDomain:
    def __init__(self):
        self.mean = None
        self.sd = None

    def fit(self, data: np.array) -> None:
        """
        Standardize the data.
        Args:
            data: 2D array
        """
        self.verify_data(data)
        self.mean = np.mean(data, axis=0)
        self.sd = np.std(data, axis=0) + 1e-5

    @staticmethod
    def verify_data(data: np.array) -> None:
        """
        Verify the training data
        Args:
            data: 2D array
        """
        if data is None:
            raise Exception("No data")
        if isinstance(data, list):
            data = np.array(data)
        if len(data.shape) != 2:
            raise Exception("2-D array is required")

    def find_outliers(self, x: np.array) -> list:
        """
        Find outliers
        Args:
            x (numpy.array): 2D array

        Returns:
            returns the indices of rows that are outliers/not in applicability domain:
        """
        self.verify_data(x)
        x = (x - self.mean) / self.sd

        max_values = np.max(x, axis=1)
        gt_three = np.where(max_values > 3.0)[0]
        min_values = np.min(x, axis=1)
        first_outliers = [i for i in gt_three if min_values[i] > 3.0]
        rest = [i for i in range(len(x)) if min_values[i] <= 3 <= max_values[i]]
        mean = np.mean(x, axis=1)
        sd = np.std(x, axis=1)
        s_new = mean + 1.28 * sd
        second_outliers = [i for (i, j) in zip(rest, s_new) if j > 3.0]

        return first_outliers + second_outliers


if __name__ == '__main__':
    sample_data = np.load(
        "redial-2020-notebook-work/valid_test_features/tpatf-3CL-balanced_randomsplit7_70_15_15_tr.npy")

    test_data = np.load("redial-2020-notebook-work/valid_test_features/tpatf-3CL-balanced_randomsplit7_70_15_15_te.npy")
    ad = ApplicabilityDomain()
    ad.fit(sample_data)
    print(f"Outliers: {ad.find_outliers(sample_data)}")
    print(f"Not in applicability domain: {ad.find_outliers(test_data)}")
