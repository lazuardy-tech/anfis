from setuptools import find_packages, setup

setup(
    name="lazuardy-anfis",
    version="0.0.1",
    description="Adaptive Neuro Fuzzy Inference System Implementation in Python.",
    url="https://github.com/lazuardy-tech/anfis",
    download_url="https://github.com/lazuardy-tech/anfis/releases",
    author="Lazuardy",
    author_email="contact@lazuardy.tech",
    license="MIT",
    package_data={
        # If any package contains *.txt files, include them:
        "": ["*.txt"]
    },
    keywords="anfis, fuzzy logic, neural networks, fnn, lazuardy",
    packages=find_packages(),
    install_requires=["numpy", "matplotlib", "scikit-fuzzy"],
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
