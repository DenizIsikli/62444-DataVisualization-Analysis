import os
import subprocess


def install_pipreqs():
    # install the following packages: pip install numpy scipy matplotlib seaborn pandas altair vega_datasets scikit-learn bokeh datashader holoviews wordcloud spacy
    subprocess.call(['pip', 'install', 'notebook', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'pandas', 'altair', 'vega_datasets', 'scikit-learn', 'bokeh', 'datashader', 'holoviews', 'wordcloud', 'spacy'])


if __name__ == "__main__":
    install_pipreqs()
    