import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
exec(open('ibp/version.py').read())
setuptools.setup(
    name="IBPILFM", 
    version=__versionnum__,
    author="Bo Liu & Linlin Li",
    author_email="bl226@duke.edu",
    description="Bayesian Infinite Latent Feature Models and IBP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LauBok/IBP4ILFM",
    packages=['IBP','IBP.test'],
    package_dir={'IBP': 'ibp', 'IBP.test': 'ibp/test'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Natural Language :: English",
        "Development Status :: 3 - Alpha",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering"
    ],
    python_requires='>=3.6',
)