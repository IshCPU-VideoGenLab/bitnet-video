from setuptools import setup, find_packages

setup(
    name="bitnet-video",
    version="0.1.0",
    author="Ishmael Affum Kwakye",
    description="1-bit quantization for CPU-native video generation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/IshCPU-VideoGenLab/bitnet-video",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=["torch>=2.0.0", "numpy>=1.24.0", "psutil>=5.9.0"],
    entry_points={"console_scripts": ["bitnet-video=bitnet_video.cli:main"]},
)
