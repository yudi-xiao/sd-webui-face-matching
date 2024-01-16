import launch
import os
import pkg_resources
from typing import Tuple, Optional
from urllib.request import urlretrieve

req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "requirements.txt")

models = {
    "models/yolo/selfie_multiclass_256x256.tflite": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
    "models/yolo/face_yolov8m.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt"
}


def comparable_version(version: str) -> Tuple:
    return tuple(version.split('.'))


def get_installed_version(package: str) -> Optional[str]:
    try:
        return pkg_resources.get_distribution(package).version
    except Exception:
        return None


with open(req_file) as file:
    for package in file:
        try:
            package = package.strip()
            if '==' in package:
                package_name, package_version = package.split('==')
                installed_version = get_installed_version(package_name)
                if installed_version != package_version:
                    launch.run_pip(f"install -U {package}",
                                   f"sd-webui-face-matching requirement: changing {package_name} version from {installed_version} to {package_version}")
            elif '>=' in package:
                package_name, package_version = package.split('>=')
                installed_version = get_installed_version(package_name)
                if not installed_version or comparable_version(installed_version) < comparable_version(package_version):
                    launch.run_pip(f"install -U {package}",
                                   f"sd-webui-face-matching requirement: changing {package_name} version from {installed_version} to {package_version}")
            elif not launch.is_installed(package):
                launch.run_pip(f"install {package}", f"sd-webui-face-matching requirement: {package}")
        except Exception as e:
            print(e)
            print(f'Warning: Failed to install {package}, some preprocessors may not work.')

for model in models.keys():
    if not os.path.exists(model):
        print(f"Downloading {model}...")
        urlretrieve(models[model], model)
