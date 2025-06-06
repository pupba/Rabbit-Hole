# from ComfyUI/folder_path.py https://github.com/comfyanonymous/ComfyUI
import os
import logging
import time
from collections.abc import Collection

folder_names_and_paths: dict[str, tuple[list[str], set[str]]] = {}
supported_pt_extensions: set[str] = {
    ".ckpt",
    ".pt",
    ".pt2",
    ".bin",
    ".pth",
    ".safetensors",
    ".pkl",
    ".sft",
}
base_path = os.getcwd()

models_dir = os.path.join(base_path, "models")
folder_names_and_paths["checkpoints"] = (
    [os.path.join(models_dir, "checkpoints")],
    supported_pt_extensions,
)
folder_names_and_paths["configs"] = ([os.path.join(models_dir, "configs")], [".yaml"])

folder_names_and_paths["loras"] = (
    [os.path.join(models_dir, "loras")],
    supported_pt_extensions,
)
folder_names_and_paths["vae"] = (
    [os.path.join(models_dir, "vae")],
    supported_pt_extensions,
)
folder_names_and_paths["text_encoders"] = (
    [os.path.join(models_dir, "text_encoders"), os.path.join(models_dir, "clip")],
    supported_pt_extensions,
)
folder_names_and_paths["diffusion_models"] = (
    [os.path.join(models_dir, "unet"), os.path.join(models_dir, "diffusion_models")],
    supported_pt_extensions,
)
folder_names_and_paths["clip_vision"] = (
    [os.path.join(models_dir, "clip_vision")],
    supported_pt_extensions,
)
folder_names_and_paths["style_models"] = (
    [os.path.join(models_dir, "style_models")],
    supported_pt_extensions,
)
folder_names_and_paths["embeddings"] = (
    [os.path.join(models_dir, "embeddings")],
    supported_pt_extensions,
)
folder_names_and_paths["diffusers"] = (
    [os.path.join(models_dir, "diffusers")],
    ["folder"],
)
folder_names_and_paths["vae_approx"] = (
    [os.path.join(models_dir, "vae_approx")],
    supported_pt_extensions,
)

folder_names_and_paths["controlnet"] = (
    [os.path.join(models_dir, "controlnet"), os.path.join(models_dir, "t2i_adapter")],
    supported_pt_extensions,
)
folder_names_and_paths["gligen"] = (
    [os.path.join(models_dir, "gligen")],
    supported_pt_extensions,
)

folder_names_and_paths["upscale_models"] = (
    [os.path.join(models_dir, "upscale_models")],
    supported_pt_extensions,
)

folder_names_and_paths["custom_nodes"] = (
    [os.path.join(base_path, "custom_nodes")],
    set(),
)

folder_names_and_paths["hypernetworks"] = (
    [os.path.join(models_dir, "hypernetworks")],
    supported_pt_extensions,
)

folder_names_and_paths["photomaker"] = (
    [os.path.join(models_dir, "photomaker")],
    supported_pt_extensions,
)

folder_names_and_paths["classifiers"] = (
    [os.path.join(models_dir, "classifiers")],
    {""},
)

output_directory = os.path.join(os.getcwd())
input_directory = os.path.join(os.getcwd())
temp_directory = os.path.join(os.getcwd(), "examples")
user_directory = os.path.join(os.getcwd())


def set_output_directory(output_dir: str) -> None:
    global output_directory
    output_directory = output_dir


def set_temp_directory(temp_dir: str) -> None:
    global temp_directory
    temp_directory = temp_dir


def set_input_directory(input_dir: str) -> None:
    global input_directory
    input_directory = input_dir


def get_output_directory() -> str:
    global output_directory
    return output_directory


def get_temp_directory() -> str:
    global temp_directory
    return temp_directory


def get_input_directory() -> str:
    global input_directory
    return input_directory


def get_user_directory() -> str:
    return user_directory


def set_user_directory(user_dir: str) -> None:
    global user_directory
    user_directory = user_dir


def get_full_path_or_raise(folder_name: str, filename: str) -> str:
    full_path = get_full_path(folder_name, filename)
    if full_path is None:
        raise FileNotFoundError(
            f"Model in folder '{folder_name}' with filename '{filename}' not found."
        )
    return full_path


def get_full_path(folder_name: str, filename: str) -> str | None:
    global folder_names_and_paths
    folder_name = map_legacy(folder_name)
    if folder_name not in folder_names_and_paths:
        return None
    folders = folder_names_and_paths[folder_name]
    filename = os.path.relpath(os.path.join("/", filename), "/")
    for x in folders[0]:
        full_path = os.path.join(x, filename)
        if os.path.isfile(full_path):
            return full_path
        elif os.path.islink(full_path):
            logging.warning(
                "WARNING path {} exists but doesn't link anywhere, skipping.".format(
                    full_path
                )
            )

    return None


def map_legacy(folder_name: str) -> str:
    legacy = {"unet": "diffusion_models", "clip": "text_encoders"}
    return legacy.get(folder_name, folder_name)


def get_folder_paths(folder_name: str) -> list[str]:
    folder_name = map_legacy(folder_name)
    return folder_names_and_paths[folder_name][0][:]


def get_save_image_path(
    filename_prefix: str, output_dir: str, image_width=0, image_height=0
) -> tuple[str, str, int, str, str]:
    def map_filename(filename: str) -> tuple[int, str]:
        prefix_len = len(os.path.basename(filename_prefix))
        prefix = filename[: prefix_len + 1]
        try:
            digits = int(filename[prefix_len + 1 :].split("_")[0])
        except:
            digits = 0
        return digits, prefix


# determine base_dir rely on annotation if name is 'filename.ext [annotation]' format
# otherwise use default_path as base_dir
def annotated_filepath(name: str) -> tuple[str, str | None]:
    if name.endswith("[output]"):
        base_dir = get_output_directory()
        name = name[:-9]
    elif name.endswith("[input]"):
        base_dir = get_input_directory()
        name = name[:-8]
    elif name.endswith("[temp]"):
        base_dir = get_temp_directory()
        name = name[:-7]
    else:
        return name, None

    return name, base_dir


def get_annotated_filepath(name: str, default_dir: str | None = None) -> str:
    name, base_dir = annotated_filepath(name)

    if base_dir is None:
        if default_dir is not None:
            base_dir = default_dir
        else:
            base_dir = get_input_directory()  # fallback path

    return os.path.join(base_dir, name)


def recursive_search(
    directory: str, excluded_dir_names: list[str] | None = None
) -> tuple[list[str], dict[str, float]]:
    if not os.path.isdir(directory):
        return [], {}

    if excluded_dir_names is None:
        excluded_dir_names = []

    result = []
    dirs = {}

    # Attempt to add the initial directory to dirs with error handling
    try:
        dirs[directory] = os.path.getmtime(directory)
    except FileNotFoundError:
        logging.warning(f"Warning: Unable to access {directory}. Skipping this path.")

    logging.debug("recursive file list on directory {}".format(directory))
    dirpath: str
    subdirs: list[str]
    filenames: list[str]

    for dirpath, subdirs, filenames in os.walk(
        directory, followlinks=True, topdown=True
    ):
        subdirs[:] = [d for d in subdirs if d not in excluded_dir_names]
        for file_name in filenames:
            try:
                relative_path = os.path.relpath(
                    os.path.join(dirpath, file_name), directory
                )
                result.append(relative_path)
            except:
                logging.warning(
                    f"Warning: Unable to access {file_name}. Skipping this file."
                )
                continue

        for d in subdirs:
            path: str = os.path.join(dirpath, d)
            try:
                dirs[path] = os.path.getmtime(path)
            except FileNotFoundError:
                logging.warning(
                    f"Warning: Unable to access {path}. Skipping this path."
                )
                continue
    logging.debug("found {} files".format(len(result)))
    return result, dirs


def filter_files_extensions(
    files: Collection[str], extensions: Collection[str]
) -> list[str]:
    return sorted(
        list(
            filter(
                lambda a: os.path.splitext(a)[-1].lower() in extensions
                or len(extensions) == 0,
                files,
            )
        )
    )


def get_filename_list_(folder_name: str) -> tuple[list[str], dict[str, float], float]:
    folder_name = map_legacy(folder_name)
    global folder_names_and_paths
    output_list = set()
    folders = folder_names_and_paths[folder_name]
    output_folders = {}
    for x in folders[0]:
        files, folders_all = recursive_search(x, excluded_dir_names=[".git"])
        output_list.update(filter_files_extensions(files, folders[1]))
        output_folders = {**output_folders, **folders_all}

    return sorted(list(output_list)), output_folders, time.perf_counter()


class CacheHelper:
    """
    Helper class for managing file list cache data.
    """

    def __init__(self):
        self.cache: dict[str, tuple[list[str], dict[str, float], float]] = {}
        self.active = False

    def get(self, key: str, default=None) -> tuple[list[str], dict[str, float], float]:
        if not self.active:
            return default
        return self.cache.get(key, default)

    def set(self, key: str, value: tuple[list[str], dict[str, float], float]) -> None:
        if self.active:
            self.cache[key] = value

    def clear(self):
        self.cache.clear()

    def __enter__(self):
        self.active = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.active = False
        self.clear()


cache_helper = CacheHelper()
filename_list_cache: dict[str, tuple[list[str], dict[str, float], float]] = {}


def cached_filename_list_(
    folder_name: str,
) -> tuple[list[str], dict[str, float], float] | None:
    strong_cache = cache_helper.get(folder_name)
    if strong_cache is not None:
        return strong_cache

    global filename_list_cache
    global folder_names_and_paths
    folder_name = map_legacy(folder_name)
    if folder_name not in filename_list_cache:
        return None
    out = filename_list_cache[folder_name]

    for x in out[1]:
        time_modified = out[1][x]
        folder = x
        if os.path.getmtime(folder) != time_modified:
            return None

    folders = folder_names_and_paths[folder_name]
    for x in folders[0]:
        if os.path.isdir(x):
            if x not in out[1]:
                return None

    return out


def get_filename_list(folder_name: str) -> list[str]:
    folder_name = map_legacy(folder_name)
    out = cached_filename_list_(folder_name)
    if out is None:
        out = get_filename_list_(folder_name)
        global filename_list_cache
        filename_list_cache[folder_name] = out
    cache_helper.set(folder_name, out)
    return list(out[0])


# def get_filename_list(
#     folder: str, allowed_exts: set = {".safetensors", ".pt", ".bin"}
# ) -> list[str]:
#     folder = os.path.join(base_path, "models", folder)
#     if not os.path.isdir(folder):
#         return []

#     return [
#         f
#         for f in os.listdir(folder)
#         if os.path.isfile(os.path.join(folder, f))
#         and os.path.splitext(f)[-1].lower() in allowed_exts
#     ]
