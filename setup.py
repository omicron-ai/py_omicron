from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="omicron",
    version="0.0.2",
    author="Ivan Leon",
    author_email="leoni@rpi.edu",
    description="An agent framework for training semantically-driven conversational agents.",
    long_description=long_description,
    url="https://github.com/omicron-ai/omicron",
    packages=find_packages(),

    install_requires=[
        "nltk", 
        "stanza"
    ],

    keywords="ai, agent",
    project_urls={
        "Source Code": "https://github.com/omicron-ai/omicron",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
