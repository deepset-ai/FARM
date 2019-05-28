"""
Simple check list from AllenNLP repo: https://github.com/allenai/allennlp/blob/master/setup.py

To create the package for pypi.

1. Change the version in __init__.py and setup.py.

2. Commit these changes with the message: "Release: VERSION"

3. Add a tag in git to mark the release: "git tag VERSION -m'Adds tag VERSION for pypi' "
   Push the tag to git: git push --tags origin master

4. Build both the sources and the wheel. Do not change anything in setup.py between
   creating the wheel and the source distribution (obviously).

   For the wheel, run: "python setup.py bdist_wheel" in the top level allennlp directory.
   (this will build a wheel for the python version you use to build it - make sure you use python 3.x).

   For the sources, run: "python setup.py sdist"
   You should now have a /dist directory with both .whl and .tar.gz source versions of allennlp.

5. Check that everything looks correct by uploading the package to the pypi test server:

   twine upload dist/* -r pypitest
   (pypi suggest using twine as other methods upload files via plaintext.)

   Check that you can install it in a virtualenv by running:
   pip install -i https://testpypi.python.org/pypi allennlp

6. Upload the final version to actual pypi:
   twine upload dist/* -r pypi

7. Copy the release notes from RELEASE.md to the tag in github once everything is looking hunky-dory.

"""
from io import open
from setuptools import find_packages, setup

setup(
    name="opensesame",
    version="0.1",
    author="Malte Pietsch, Timo Moeller, Branden Chan, Huggingface Team Authors, Google AI Language Team Authors, Open AI team Authors",
    author_email="malte.pietsch@deepset.ai",
    description="Toolkit for pretraining, finetuning and evaluating transformer based language models",
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    keywords='BERT NLP deep learning languagemodel transformer',
    license='Apache',
    url="https://gitlab.com/deepset-ai/ml/lm/opensesame",
    packages=find_packages(exclude=["*.tests", "*.tests.*",
                                    "tests.*", "tests"]),
    install_requires=['torch>=0.4.1',
                      'numpy',
                      'boto3',
                      'requests',
                      'tqdm',
                      'regex'],
    entry_points={
      'console_scripts': [
        "pytorch_pretrained_bert=pytorch_pretrained_bert.__main__:main",
      ]
    },
    # python_requires='>=3.5.0',
    tests_require=['pytest'],
    classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
