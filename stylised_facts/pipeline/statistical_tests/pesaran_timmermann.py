import numpy as np
from scipy.stats import norm


class PesaranTimmermann:
    # Note here that we take in returns. Not values
    def __init__(self, actual_returns, forecast_returns):
        self._actual_returns = np.array(actual_returns)
        self._forecast_returns = np.array(forecast_returns)

        self.t_stat = None
        self.p_value = None

    def get_t_stat(self):
        # pyz = np.mean(np.sign(self._actual_returns) == np.sign(self._forecast_returns))
        pyz = np.mean(
            np.where(
                (((self._actual_returns > 0) & (self._forecast_returns > 0))
            | ((self._actual_returns < 0) & (self._forecast_returns < 0))), 1, 0))
        py = np.mean(np.where(self._actual_returns > 0, 1, 0))
        pz = np.mean(np.where(self._forecast_returns > 0, 1, 0))
        qy = py * (1 - py) / len(self._forecast_returns)
        qz = pz * (1 - pz) / len(self._forecast_returns)
        p = pz * py + (1 - pz) * (1 - py)
        v = p * (1 - p) / len(self._forecast_returns)
        w = qz * (2 * py - 1) ** 2 + qy * (2 * pz - 1) ** 2 + 4 * qy * qz
        self.t_stat = (pyz - p) / np.sqrt(v - w)
        return self

    def get_p_value(self):
        self.p_value = 1 - norm.cdf(self.t_stat)
        return self

    def run_hypothesis_test(self, alpha):
        if self.p_value < alpha:
            print(f"Reject null hypothesis at confidence level of {1 - alpha} "
                  f"with p_value = {self.p_value:.4f}")
        else:
            print(f"Failed to reject null hypothesis at confidence level of {100 * (1- alpha)}% "
                  f"with p_value = {self.p_value:.4f}")


if __name__ == '__main__':
    test_actual_returns = [0.01, -0.93, 0.24, 0.68, 0.04, -0.34, -0.63, -0.59, -0.80, 0.07, -0.27, -0.27, -0.31, 0.38, 0.46]
    test_forecast_returns = [0.78, 0.12, 0.02, 0.68, -0.83, -0.48, 0.37, -0.20, -0.44, 0.38, 0.96, -0.37, -0.96, 0.31, 0.26]

    pesaran_timmermann = PesaranTimmermann(actual_returns=test_actual_returns,
                                           forecast_returns=test_forecast_returns
                                           )
    pesaran_timmermann. \
        get_t_stat(). \
        get_p_value(). \
        run_hypothesis_test(alpha=0.01)
