import json
from pathlib import Path


dict_template = {
    "neus_config": "./confs/thin_structure.conf",
    # target neus
    "target_ckpt": "path/to/target_ckpt",
    "target_case": "furina_target",

    # source neus
    "source_neus": ["path/to/source_neus1", "path/to/source_neus2"],
    "source_case": ["furina_source1", "furina_source2"],
}

def generate_json_template():
    with open("./confs/json/neus_reg_config.json", "w") as f:
        json.dump(dict_template, f, indent=4)

generate_json_template()
print("JSON template generated successfully!")