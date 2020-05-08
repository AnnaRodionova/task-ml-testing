import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ml-testing",
    version="0.0.1",
    author="Anna Rodionova",
    author_email="sghtarbs@gmail.com",
    description="Package for model validation testing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=['ml_testing', 'ml_testing.classification_tests', 'ml_testing.regression_tests', 'ml_testing.plotting'],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=["sklearn", "matplotlib"],
)