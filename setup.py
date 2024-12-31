from setuptools import setup, find_packages

# Open and read the contents of the README file
with open('README.md', 'r', encoding='utf-8') as readme_file:
    long_description = readme_file.read()

setup(
    name="pulpo-dev",
    version="0.1.3",
    description="Pulpo package for optimization in LCI databases",
    author="Fabian Lechtenberg",
    author_email="fabian.lechtenberg@chem.ethz.ch",
    url="https://github.com/flechtenberg/pulpo",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.10, <3.13",
    install_requires=[
        "fs==2.4.16",
        "pyomo<=6.6.2",
        "highspy==1.8.0",
        "ipython==8.14.0",
        "jupyterlab",
        "numpy<2.0.0",
        "pandas",
        "tqdm",
        "xlsxwriter",
    ],
    extras_require={
        "bw2": [
            "bw2calc<=1.8.2",
            "bw2data<=3.9.9",
        ],
        "bw25": [
            "bw2calc>=2.0.0",
            "bw2data>=4.0.0",
        ],
    },
    long_description=long_description,
    long_description_content_type='text/markdown',  # Specify the content type of the README
)
