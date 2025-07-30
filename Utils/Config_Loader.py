import os

def load_config(config_path=None):
    possible_paths = []

    if config_path:
        possible_paths.append(config_path)

    possible_paths.append(os.path.join(os.getcwd(), "config.txt"))

    this_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths.append(os.path.join(this_dir, "config.txt"))

    # Project root (one level up from this file)
    project_root = os.path.dirname(this_dir)
    possible_paths.append(os.path.join(project_root, "config.txt"))

    # Parent of project root (two levels up)
    parent_of_project_root = os.path.dirname(project_root)
    possible_paths.append(os.path.join(parent_of_project_root, "config.txt"))

    for path in possible_paths:
        if os.path.exists(path):
            config = {}
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        k, v = line.split("=", 1)
                        config[k.strip()] = v.strip()
            return config

    raise FileNotFoundError(f"Config file not found in any of: {possible_paths}")
