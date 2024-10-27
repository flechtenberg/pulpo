from setuptools import setup, find_packages

# Open and read the contents of the README file
with open('README_pypi.md', 'r', encoding='utf-8') as readme_file:
    long_description = readme_file.read()

setup(
    name="pulpo-dev",
    version="0.1.1",
    description="pulpo package for optimization in LCI databases",
    author="Fabian Lechtenberg",
    author_email="fabian.lechtenberg@upc.edu",
    url="https://github.com/flechtenberg/pulpo",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.10, <=3.12",
    install_requires=[
        "bw2data==3.6.6",
        "fs==2.4.16",
        "pyomo<=6.6.2",
        "highspy==1.8.0",
        "ipython==8.14.0",
        "jupyterlab",
        "tqdm",
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',  # Specify the content type of the README
)
