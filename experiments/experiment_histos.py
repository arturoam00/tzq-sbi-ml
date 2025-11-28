from pathlib import Path

from madminer.limits import AsymptoticLimits

from .base.base_experiment import BaseExperiment
from .limits.asymptotic_limits import Limits
from .plotting import plot_llr


class ExperimentHistos(BaseExperiment):

    # I pass `model=<model object>` to the experiment
    # constructor in `run.py`. Allow for it here
    def __init__(self, *args, **kwds):
        kwds["key"] = "histos"
        super().__init__(*args, **kwds)

    def _init(self):
        pass

    def eval_lims(self):
        alims = AsymptoticLimits(self.cfg.dataset.events_file)
        grid, pvalues, mle, llr_kin, rate_ll, *_ = alims.expected_limits(
            mode="histo",
            hist_vars=self.cfg.limits.hist_vars,
            theta_true=self.cfg.limits.asimov.theta_true,
            grid_ranges=self.cfg.limits.theta_ranges,
            grid_resolutions=self.cfg.limits.resolutions,
            luminosity=self.cfg.limits.luminosity,
            sample_only_from_closest_benchmark=self.cfg.limits.asimov.sample_only_from_closest_benchmark,
            n_histo_toys=self.cfg.limits.n_toys,
            test_split=self.cfg.limits.test_split,
        )
        return Limits(
            list(alims.parameters.keys()), grid, pvalues, mle, llr_kin, rate_ll
        )

    def plot_llr(self, to=None):
        return plot_llr([self.checkpoints.limits], labels=[str(self)], to=to)

    def plot(self):
        self.plot_llr(Path(self.cfg.data.run_dir) / "llr.png")

    def _run(self):
        if self.cfg.modes.eval:
            self.checkpoints.limits = self.eval_lims()
        if self.cfg.modes.plot:
            self.plot()
