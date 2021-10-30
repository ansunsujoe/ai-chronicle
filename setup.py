import setuptools

def readme():
    with open('README.md') as f:
        return f.read()

def requirements():
    with open("requirements.txt") as f:
        return f.read().split("\n")

setuptools.setup(
    name="dash",
    version="1.0.0",
    long_description=readme(),
    packages=setuptools.find_packages(),
    install_requires=requirements(),
    python_requires=">=3.7"
)