from setuptools import setup, find_packages


setup(
   name='plum',
   version='0.1',
   description='A less painful way to run neural network experiments.',
   author='Chris Kedzie',
   author_email='kedzie@cs.columbia.edu',
   packages=find_packages(),
   dependency_links = [],
   include_package_data=True,
   zip_safe=False,
   install_requires = ["jsonnet", "ujson", "matplotlib", "pandas", 
                       "torch==2.2.0",
                       "tb-nightly", "future", "scikit-learn", "Pillow"],
   entry_points={"console_scripts": ["plumr=plum.plumr:main"]},
   package_data={
       'plum': ['jsonnet/*'],
   },
)

