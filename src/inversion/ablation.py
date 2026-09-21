"""Explicit switch matrix for named baselines and one-component removals."""
import copy

VARIANTS = {
    "latent_ga": {"local":False,"bank":False,"dynamic":False,"qc":True},
    "ga_local": {"local":True,"bank":False,"dynamic":True,"qc":True},
    "ga_bank_static": {"local":False,"bank":True,"dynamic":False,"qc":True},
    "full": {"local":True,"bank":True,"dynamic":True,"qc":True},
    "no_local": {"local":False,"bank":True,"dynamic":True,"qc":True},
    "no_bank": {"local":True,"bank":False,"dynamic":True,"qc":True},
    "no_dynamic": {"local":True,"bank":True,"dynamic":False,"qc":True},
    "no_qc": {"local":True,"bank":True,"dynamic":True,"qc":False},
}


def variant_config(config: dict,name: str) -> dict:
    if name not in VARIANTS:
        raise ValueError(f"Unknown ablation {name}; choose {list(VARIANTS)}")
    result = copy.deepcopy(config)
    switches = VARIANTS[name]
    result["local_fwi"]["enabled"] = switches["local"]
    result["zbank"]["enabled"] = switches["bank"]
    result["dynamic_environment"] = switches["dynamic"]
    result["qc"]["enabled"] = switches["qc"]
    return result
