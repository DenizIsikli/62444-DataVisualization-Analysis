import os
import subprocess


def install_pipreqs():
    # Install the following packages:
    # numpy scipy matplotlib seaborn pandas altair vega_datasets scikit-learn bokeh datashader holoviews wordcloud spacy
    subprocess.call(['pip', 'install', 'notebook', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'pandas', 'altair', 'vega_datasets', 'scikit-learn', 'bokeh', 'datashader', 'holoviews', 'wordcloud', 'spacy', 'pyarrow', 'fastparquet', 'plotly'])


if __name__ == "__main__":
    install_pipreqs()
    