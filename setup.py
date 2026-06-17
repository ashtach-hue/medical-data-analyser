from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='medical-data-analyser',
    version='0.1.0',
    author='ashtach-hue',
    description='Healthcare analytics platform with ML disease prediction',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ashtach-hue/medical-data-analyser',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Healthcare Industry',
        'Topic :: Scientific/Engineering :: Medical Science Apps',
    ],
    python_requires='>=3.9',
    install_requires=[
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'scikit-learn>=1.3.0',
        'scipy>=1.11.0',
    ],
)
