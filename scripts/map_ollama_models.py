# Based on https://gist.github.com/bsharper/03324debaa24b355d6040b8c959bc087

# Run without arguments to see what is found.
# Run with a path as an argument to create links to ollama models there.
# This will remove any files that follow the exact filename as the new link file, so use with caution!

# Example usage: python map_ollama_models.py ~/llama.cpp/models
# Or with custom Ollama path: python map_ollama_models.py ~/llama.cpp/models --ollama_models_path /path/to/models

import os
import sys
import json
import platform
import argparse

def get_ollama_model_path():
    # Check if OLLAMA_MODELS environment variable is set
    env_model_path = os.environ.get('OLLAMA_MODELS')
    if env_model_path:
        return env_model_path

    # Determine the model path based on the operating system
    system = platform.system()
    if system == 'Darwin':  # macOS
        return os.path.join(os.path.expanduser('~'), '.ollama', 'models')
    elif system == 'Linux':
        return '/usr/share/ollama/.ollama/models'
    elif system == 'Windows':
        return os.path.join(os.environ['USERPROFILE'], '.ollama', 'models')
    else:
        raise Exception('Unsupported Operating System')

def extract_digests(json_record):
    result = {}
    for layer in json_record.get('layers', []):
        media_type = layer.get('mediaType')
        digest = layer.get('digest')
        if media_type == "application/vnd.ollama.image.model":
            result['gguf'] = digest
        elif media_type == "application/vnd.ollama.image.projector":
            result['mmproj'] = digest
    return result

def search_for_models(directory):
    models = {}
    models_map = {}
    blobs = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            sz = os.path.getsize(full_path)
            if root.endswith("blobs"):
                blobs[file.replace("sha256-", "sha256:")] = full_path
            if (file == "latest" or ("registry.ollama.ai" in full_path and "library" in full_path)) and sz < 2048:
                els = full_path.split(os.sep)
                model_name = f"{els[-2]}-{els[-1]}"
                j = json.load(open(full_path))
                models[model_name] = extract_digests(j)
    for model_name in models.keys():
        model = models[model_name]
        obj = model
        try:
            obj = {x: blobs[model[x]] for x in model.keys()}
        except KeyError as ex:
            print(ex)
            print("This likely means that a manifest is pointing to a file that does not exist")
        models_map[model_name] = obj
    return models_map

def generate_link_pairs(models_map, target_path=""):
    if target_path:
        target_path = os.path.expanduser(target_path)
    links = []
    for model_name in models_map:
        model = models_map[model_name]
        for file_type in model:
            file_path = model[file_type]
            filename = f"{model_name}.{file_type}"
            if target_path:
                filename = os.path.join(target_path, filename)
            links.append({'target': file_path, 'linkpath': filename})
    return links

def print_link_script(links):
    is_windows = platform.system() == "Windows"
    for link in links:
        if is_windows:
            print(f"mklink '{link['linkpath']}' '{link['target']}'")
        else:
            print(f"ln -s '{link['target']}' '{link['linkpath']}'")

def create_links(links):
    for link in links:
        linkpath = link['linkpath']
        target = link['target']
        print(f'Creating link "{linkpath}" => "{target}"')
        if not os.path.exists(target):
            print(f'Skipping "{target}", file does not exist')
            continue
        if os.path.exists(linkpath) or os.path.islink(linkpath):
            os.unlink(linkpath)
        os.symlink(target, linkpath)

def header(ollama_path, link_path=""):
    width = 60
    print("=" * width)
    print(f"Ollama models path : {ollama_path}")
    if link_path:
        link_path = os.path.expanduser(link_path)
        print(f"Link path          : {link_path}")
    print("=" * width)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates symbolic links to the models downloaded by ollama"
    )
    parser.add_argument(
        "output_path",
        help="Path where symbolic links will be created"
    )
    parser.add_argument(
        "--ollama_models_path",
        help="Path to ollama models directory (optional, autodetected if not provided)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        sys.exit(f'Error: provided output_path "{args.output_path}" does not exist')

    ollama_path = args.ollama_models_path or get_ollama_model_path()
    header(ollama_path, args.output_path)

    models_map = search_for_models(ollama_path)
    links = generate_link_pairs(models_map, args.output_path)
    create_links(links)
