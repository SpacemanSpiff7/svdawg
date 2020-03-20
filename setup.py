import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='svdawg',
     version='0.1',
     scripts=['svdawg.py'] ,
     author="Simone Longo",
     author_email="s.longo@utah.edu",
     description="SVD accessories, widgets, and graphics",
     install_requires=[
          'pandas',
          'numpy',
          'seaborn',
          'matplotlib',
          'sklearn'
      ]
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/SpacemanSpiff7/svdawg",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
