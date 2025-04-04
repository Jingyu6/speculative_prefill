from setuptools import find_packages, setup

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="speculative_prefill", 
    version="1.0.0", 
    description="Speculative Prefill: Speeding up LLM Inference via Token Importance Transferability. ",

    url="https://github.com/Jingyu6/speculative_prefill.git#egg=speculative_prefill", 
    author="Jingyu Liu", 
    author_email="jingyu6@uchicago.edu", 

    python_requires=">=3.10", 
    packages=find_packages(include=["speculative_prefill", "speculative_prefill.*"]), 
    install_requires=install_requires
)
