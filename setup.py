from setuptools import setup, find_packages

setup(
    name="cuif-generator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jinja2",
        "pyyaml",
        "watchdog",
    ],
    entry_points={
        'console_scripts': [
            'cuif-generate=cuif_generator.cli:main',
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool for generating CUDA-ROS2 integration files from CUIF specifications",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cuif-generator",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
) 