from io import open

from setuptools import find_packages, setup

setup(
    name="farm",
    version="0.1",
    author="Malte Pietsch, Timo Moeller, Branden Chan, Huggingface Team Authors, Google AI Language Team Authors, Open AI team Authors",
    author_email="malte.pietsch@deepset.ai",
    description="Toolkit for finetuning and evaluating transformer based language models",
    long_description=open("readme.rst", "r", encoding="utf-8").read(),
    long_description_content_type="text/x-rst",
    keywords="BERT NLP deep learning language-model transformer",
    license="Apache",
    url="https://gitlab.com/deepset-ai/ml/lm/farm",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=[
        "torch>=1.1.0",
        "numpy",
        "boto3",
        "requests",
        "tqdm",
        "mlflow>=1.0.0",
        "scipy",
        "sklearn",
        "seqeval",
    ],
    python_requires=">=3.5.0",
    tests_require=["pytest"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
