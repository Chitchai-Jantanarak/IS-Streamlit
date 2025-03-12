from setuptools import setup, find_packages

setup(
    name="intelligent-system-project",
    version="0.1.0",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'streamlit==1.32.0',
        'scikit-learn==1.4.1',
        'numpy==1.26.4',
        'pandas==2.2.1',
        'matplotlib==3.8.3',
        'seaborn==0.12.2',
        'tensorflow==2.15.0',
        'keras-visualizer==2.4',
        'plotly==5.18.0',
        'ydata_profiling',
        'IPython'
    ],
)