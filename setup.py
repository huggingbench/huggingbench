from setuptools import setup, find_packages

setup(
    name='hugging-bench-project',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        # install rust compier as need for building some packages in wheel   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
        # and export PATH="$HOME/.cargo/bin:$PATH"
        'tritonclient[all]==2.33.0',
        'prometheus-client==0.16.0',
        'docker==6.1.2',
        'datasets==2.12.0',
        'transformers==4.11.3',
        'numpy',
    ]
)
