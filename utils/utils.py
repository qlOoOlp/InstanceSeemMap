import yaml

def load_config(yaml_file_path):
    try:
        with open(yaml_file_path, 'r') as file:
            config = yaml.safe_load(file)  # YAML 파일을 Python 딕셔너리로 변환
        return config
    except FileNotFoundError:
        print(f"Error: File {yaml_file_path} not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None
