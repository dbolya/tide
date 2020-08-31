import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text(encoding='utf8')

# This call to setup() does all the work
setup(
    name="tidecv",
    version="1.0.1",
    description="A General Toolbox for Identifying ObjectDetection Errors",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/dbolya/tide",
    author="Daniel Bolya",
    author_email="dbolya@gatech.edu",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6',
    packages=["tidecv", "tidecv.errors"],
    include_package_data=True,
    install_requires=["appdirs", "numpy", "pycocotools", "opencv-python", "seaborn", "pandas", "matplotlib"],
    # entry_points={
    #     "console_scripts": [
    #         "tidecv=tidecv.__main__:main",
    #     ]
    # },
)
