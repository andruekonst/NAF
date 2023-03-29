import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    install_requirements = f.read().splitlines()

setuptools.setup(
    name="naf",
    version="0.0.1",
    author="Andrei V. Konstantinov",
    author_email="andrue.konst@gmail.com",
    description="Neural Attention Forest",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/andruekonst/NAF",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=install_requirements
)
