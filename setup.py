from setuptools import setup, find_packages

setup(
    name="omicron",
    version="0.0.2",
    packages=find_packages(),

    install_requires=[
        "nltk", 
        "stanza"
    ],

    author="Ivan Leon",
    author_email="leoni@rpi.edu",
    description="An agent framework for training semantically-driven conversational agents.",
    keywords="ai, agent",
    project_urls={
        "Source Code": "https://github.com/omicron-ai/omicron",
    }
)
