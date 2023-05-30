from setuptools import setup, find_packages

# There might be a need to install rust compier as need for building some packages in wheel   
# curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# and export PATH="$HOME/.cargo/bin:$PATH"
setup(
    name='hugging-bench-project',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'tritonclient[all]==2.33.0',
        'prometheus-client==0.16.0',
        'docker==6.1.2',
        'datasets==2.12.0',
        'transformers==4.29.2',
        'torchvision==0.15.2',
        'numpy',
    ],
    python_requires='>=3.9'
)
