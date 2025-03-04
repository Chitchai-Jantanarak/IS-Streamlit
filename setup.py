from setuptools import setup, find_packages

setup(
    name="streamlit-business-dashboard",
    version="0.1.0",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'streamlit==1.32.0',
        'pandas==2.2.1',
        'plotly==5.18.0',
        'numpy==1.26.4',
        'matplotlib==3.8.3',
        'scikit-learn==1.4.1',
    ],
)