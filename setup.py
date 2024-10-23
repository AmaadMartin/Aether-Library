from setuptools import setup, find_packages

setup(
    name="aetherllm",  # The name users will use to install your package (pip install aether)
    version="0.1.5",  # Starting version number
    author="Wadih Pazos",
    author_email="wadpod7@gmail.com",
    description="A Python library for CI/CD with LLMs.",
    packages=find_packages(),  # Automatically find the _Aether package
    install_requires=open("requirements.txt")
    .read()
    .splitlines(),  # Read dependencies from requirements.txt
)