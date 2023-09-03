from hydra import compose, initialize
from sys import argv
import hydra

if hydra.core.global_hydra.GlobalHydra.instance() is not None:
    hydra.core.global_hydra.GlobalHydra.instance().clear()
initialize(config_path="config", version_base=None)
# hprams = compose(config_name="configs", overrides=argv[1:])
hprams = compose(config_name="configs")
