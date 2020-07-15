from io import open

from setuptools import find_packages, setup


def parse_requirements(filename):
    """
    Parse a requirements pip file returning the list of required packages. It exclude commented lines and --find-links directives.

    Args:
        filename: pip requirements requirements

    Returns:
        list of required package with versions constraints

    """
    with open(filename) as file:
        parsed_requirements = file.read().splitlines()
    parsed_requirements = [line.strip()
                           for line in parsed_requirements
                           if not ((line.strip()[0] == "#") or line.strip().startswith('--find-links'))]
    return parsed_requirements


def get_dependency_links(filename):
    """
     Parse a requirements pip file looking for the --find-links directive.
    Args:
        filename:  pip requirements requirements

    Returns:
        list of find-links's url
    """
    with open(filename) as file:
        parsed_requirements = file.read().splitlines()
    dependency_links = list()
    for line in parsed_requirements:
        line = line.strip()
        if line.startswith('--find-links'):
            dependency_links.append(line.split('=')[1])
    return dependency_links


dependency_links = get_dependency_links('requirements.txt')
parsed_requirements = parse_requirements('requirements.txt')

setup(
    name="farm",
    version="0.4.6",
    author="Malte Pietsch, Timo Moeller, Branden Chan, Tanay Soni",
    author_email="malte.pietsch@deepset.ai",
    description="Toolkit for finetuning and evaluating transformer based language models",
    long_description=open("readme.rst", "r", encoding="utf-8").read(),
    long_description_content_type="text/x-rst",
    keywords="BERT NLP deep learning language-model transformer qa question-answering transfer-learning",
    license="Apache",
    url="https://gitlab.com/deepset-ai/ml/lm/farm",
    download_url="https://github.com/deepset-ai/FARM/archive/0.4.5.tar.gz",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    dependency_links=dependency_links,
    install_requires=parsed_requirements,
    python_requires=">=3.6.0",
    extras_require={
        "fasttext": ["fasttext==0.9.1"],
        "onnx": ["onnxruntime"],
    },
    tests_require=["pytest"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
