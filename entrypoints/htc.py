import tempfile
from pathlib import Path

import htcondor2
from omegaconf import OmegaConf


def submit(*, description, cfg):
    schedd = htcondor2.Schedd()
    job_description = htcondor2.Submit(dict(description))

    with tempfile.NamedTemporaryFile(mode="w+", delete=True) as tmp:
        # Save config to temporary file
        OmegaConf.save(cfg, tmp)

        # Data to pass temporary file to node
        itemdata = {"input_file": tmp.name, "input_file_name": Path(tmp.name).name}

        # Schedule job
        schedd.submit(description=job_description, itemdata=iter(itemdata))
