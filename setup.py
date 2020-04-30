from setuptools import setup, find_packages

setup(
    name="omicron",
    version="0.0.1",
    packages=find_packages(),

    install_requires=[
        "nltk", 
        "stanza"
    ],

    author="Ivan Leon",
    author_email="leoni@rpi.edu",
    description="A semantically-driven language generation system.",
    keywords="nlg, nlp, language",
    project_urls={
        "Source Code": "https://github.com/ielm/omicron",
    }
)
