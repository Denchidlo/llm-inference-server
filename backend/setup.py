
from setuptools import setup 
  
setup( 
    name='backend', 
    version='0.1', 
    description='A sample Python package', 
    author='John Doe', 
    author_email='jdoe@example.com', 
    packages=['batched_inference'], 
    install_requires=[ 
        'numpy', 
        'pandas', 
    ], 
) 
