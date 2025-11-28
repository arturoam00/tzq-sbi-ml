import numpy as np

from .asymptotic_limits import AsymptoticLimits


class AsymptoticLimitsRatios(AsymptoticLimits):

    NEEDS_HISTOS = False

    def log_r_kin(self, **kwds):
        return np.mean(kwds["predictions"], axis=1)
