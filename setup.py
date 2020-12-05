import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="eeg_classifier",
    version="0.0.1",
    author="Russell Weas",
    author_email="russweas@gmail.com",
    license='BSD 2-clause',
    description="A set of tools to help design EEG classifier models in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['mne', 'pandas', 'numpy', 'scipy', 'seaborn', 'sklearn'],
    python_requires='>=3.6',
)