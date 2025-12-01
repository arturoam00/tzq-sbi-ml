import io
import tarfile
import tempfile
from pathlib import Path

import htcondor2
from omegaconf import OmegaConf


def _tarball(cfg):
    input_dir = Path("inputs")
    input_dir.mkdir(exist_ok=True, parents=True)

    # Create unique safe name for tarball
    with tempfile.NamedTemporaryFile(mode="r+", dir=input_dir, suffix=".tar.gz") as tmp:
        tarball = Path(tmp.name)

    with tarfile.open(tarball, "w:gz") as tar:
        # Add data and possibly model checkpoints
        tar.add(cfg.dataset.path)
        if Path(cfg.data.run_dir).exists():
            tar.add(cfg.data.run_dir)

        # Add configuration object
        cfg_encoded = OmegaConf.to_yaml(cfg).encode()
        info = tarfile.TarInfo(name="input.yaml")
        info.size = len(cfg_encoded)
        tar.addfile(info, io.BytesIO(cfg_encoded))

    return tarball


def submit(*, description, description_addition={}, cfg):
    # Prepare job description
    job_description = htcondor2.Submit(dict(description))
    job_description.update(description_addition)

    tb = _tarball(cfg)
    itemdata = [{"tarball": str(tb), "tarball_name": tb.name}]

    # Schedule job
    schedd = htcondor2.Schedd()
    schedd.submit(description=job_description, itemdata=iter(itemdata))
