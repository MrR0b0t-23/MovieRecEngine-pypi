import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MovieRecEngine", 
    version="0.1.0",
    author="Ashwin",
    author_email="imashwin02@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MrR0b0t-23/MovieRecEngine",
    project_urls={
        "Bug Tracker": "https://github.com/MrR0b0t-23/MovieRecEngine/issues",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords = ['Movie Recommendation Engine', 'Machine Learning', 
    'Movie Recommendation System', 'Collabative Filtering', 'Recommendation', 'MovieRecEngine','Pytorch Recommendation' ],
    packages=["MovieRecEngine"],
    include_package_data=True,
    install_requires=["torch>=1.6.0", "tez>=0.1.2","pandas>=1.2.2"],
    python_requires=">=3.6",
)