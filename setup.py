from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="lazuardy_anfis",
    version="0.0.2",
    description="Adaptive Neuro Fuzzy Inference System Implementation in Python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Lazuardy",
    author_email="contact@lazuardy.tech",
    packages=find_packages(
        include=[
            "anfis",
            "fuzzy-logic",
            "neural-networks",
            "fnn",
            "lazuardy",
            "lazuardy_anfis",
        ]
    ),
    keywords="anfis, fuzzy logic, neural networks, fnn, lazuardy, lazuardy anfis",
    url="https://github.com/lazuardy-tech/anfis",
    download_url="https://github.com/lazuardy-tech/anfis/releases",
    license="MIT",
    install_requires=required,
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
    ],
)
