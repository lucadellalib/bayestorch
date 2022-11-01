# ==============================================================================
# Copyright 2022 Luca Della Libera.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Setup script."""


################## Install setup requirements ##################
def _preinstall_requirement(requirement, options=None):
    import subprocess

    args = ["pip", "install", requirement, *(options or [])]
    return_code = subprocess.call(args)
    if return_code != 0:
        raise RuntimeError(f"{requirement} installation failed")


for requirement in ["setuptools>=58.0.4", "wheel>=0.37.1"]:
    _preinstall_requirement(requirement)
################################################################


import os  # noqa: E402

from setuptools import find_packages, setup  # noqa: E402


_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(_ROOT_DIR, "bayestorch", "version.py")) as f:
    tmp = {}
    exec(f.read(), tmp)
    _VERSION = tmp["VERSION"]

with open(os.path.join(_ROOT_DIR, "README.md"), encoding="utf-8") as f:
    _README = f.read()

setup(
    name="bayestorch",
    version=_VERSION,
    description="Bayesian deep learning library for fast prototyping based on PyTorch",
    long_description=_README,
    long_description_content_type="text/markdown",
    author="Luca Della Libera",
    author_email="luca.dellalib@gmail.com",
    url="https://github.com/lucadellalib/bayestorch",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    license="Apache License 2.0",
    keywords=["Bayesian deep learning", "PyTorch"],
    platforms=["OS Independent"],
    include_package_data=True,
    install_requires=["torch>=1.6.0"],
    extras_require={
        "test": [
            "pytest>=5.4.3",
            "pytest-cov>=2.9.0",
        ],
        "dev": [
            "black>=22.3.0",
            "cibuildwheel>=2.3.1",
            "flake8>=3.8.3",
            "flake8-bugbear>=20.1.4",
            "isort>=5.4.2",
            "pre-commit>=2.6.0",
            "pre-commit-hooks>=3.2.0",
            "pytest>=5.4.3",
            "pytest-cov>=2.9.0",
            "twine>=3.3.0",
        ],
    },
    python_requires=">=3.6",
)
