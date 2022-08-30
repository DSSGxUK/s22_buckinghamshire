import yaml
import inspect
import re

from src import params

if __name__ == "__main__":
    # This will ensure that pyyaml expands all nested variables
    yaml.Dumper.ignore_aliases = lambda *args: True

    param_dict = {
        item: val
        for item, val in params.__dict__.items()
        if not (
            item.startswith("_") or inspect.ismodule(val) or inspect.isfunction(val)
        )
    }
    preamble_str = """# This file was generated by scripts/generate_params.py"""
    with open(params.PARAMS_YAML, "w") as p:
        print(preamble_str, file=p)
        print(re.sub(r" ?!!python/.*\n", "\n", yaml.dump(param_dict)), file=p)
