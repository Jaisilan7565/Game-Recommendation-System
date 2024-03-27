from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_discription = fh.read()

AUTHOR_NAME="Vishu Poddar"
SRC_REPO = 'src'
LIST_OF_REQUIREMENTS = ['streamlit']

setup(
    name = SRC_REPO,
    version='0.0.1',
    author=AUTHOR_NAME,
    description="A small ML Project",
    long_discription= long_discription,
    long_discription_content_type="text/markdown",
    package=[SRC_REPO],
    python_requires= '>=3.7',
    install_requires= LIST_OF_REQUIREMENTS,

)