import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MovieRecSys", 
    version="0.0.3",
    author="Ashwin",
    author_email="imashwin02@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MrR0b0t-23/MovieRecSys.git",
    project_urls={
        "Bug Tracker": "https://github.com/MrR0b0t-23/MovieRecSys/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords = ['Movie', 'Recommendation System', 'Machine Learning', 'Movie Recommendation System', 'Collabative Filtering', 'Movie Recommendation Engine'],
    include_package_data=True,
    install_requires=["torch>=1.6.0", "numpy==1.20.2","tez==0.1.2","pandas==1.2.3","scikit_learn==0.24.1"],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)