from setuptools import find_packages, setup

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="XXX", 
    version="1.0.0", 
    description="Speculative Prefill: Speeding up LLM Inference via Token Importance Transferability. ",

    url="XXX", 
    author="XXX XXX", 
    author_email="XXX", 

    python_requires=">=3.10", 
    packages=find_packages(include=["speculative_prefill", "speculative_prefill.*"]), 
    install_requires=install_requires
)
