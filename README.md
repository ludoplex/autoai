<a href="https://pix.blobcity.com/I1Nk23FY"><img src="https://blobcity.com/assets/img/blobcity-logo.svg" style="width: 40%"/></a>

[![PyPI version](https://badge.fury.io/py/blobcity.svg)](https://badge.fury.io/py/blobcity)
![Downloads](https://shields.io/pypi/dm/blobcity)
[![Python](https://shields.io/pypi/pyversions/blobcity)](https://pypi.org/project/blobcity/)
[![License](https://shields.io/pypi/l/blobcity)](https://pypi.org/project/blobcity/)

[![Contributors](https://shields.io/github/contributors/blobcity/autoai)](https://github.com/blobcity/autoai)
[![Commit Activity](https://shields.io/github/commit-activity/m/blobcity/autoai)](https://github.com/blobcity/autoai)
[![Last Commit](https://shields.io/github/last-commit/blobcity/autoai)](https://github.com/blobcity/autoai)
[![Slack](https://shields.io/badge/join%20discussion-slack-orange)](https://pix.blobcity.com/E2Bepr4w)

[![GitHub Stars](https://shields.io/github/stars/blobcity?style=social)](https://github.com/blobcity)
[![Twitter](https://shields.io/twitter/follow/blobcity?label=Follow)](https://twitter.com/blobcity)


# BlobCity AutoAI
A framework to find, train and generate code for the best performing AI model. Works on Classification and Regression problems on numerical data. This is a beta release. The framework is being actively worked upon. Please report any issues you encounter.

[![Issues](https://shields.io/github/issues/blobcity/autoai)](https://github.com/blobcity/autoai/issues)


# Getting Started
``` shell
pip install blobcity
```

``` Python
import blobcity as bc
model = bc.train(file="data.csv", target="Y_column")
model.spill("my_code.py")
```
`Y_column` is the name of the target column. The column must be present within the data provided. 

Automatic inference of Regression / Classification is supported by the framework.

Data input formats supported include:
1. Local CSV / XLSX file
2. URL to a CSV / XLSX file
3. Pandas DataFrame 

``` Python
model = bc.train(file="data.csv", target="Y_column") #local file
```

``` Python
model = bc.train(df=my_df, target="Y_column") #DataFrame
```

``` Python
model = bc.train(file="https://example.com/data.csv", target="Y_column") #url
```

# Code Generation
The `spill` function generates the model code with exhaustive documentation. Training code is also included for scikit-learn models. TensorFlow and other DNN models produce only the test / final use code. 


Multiple formats of code generation is supported by the framework. The `spill` function can be used to generate both `ipynb` and `py` files. You can choose to enable or disable code export along with docs.

``` Python
model.spill("my_code.ipynb"); #produces Jupyter Notebook file with full markdown docs

model.spill("my_code.py") #produces python code with minimal docs

model.spill("my_code.py", docs=True) #python code with full docs

model.spill("my_code.ipynb", docs=False) #Notebook file with minimal markdown
```

# Predictions
Use a trained model to generate predictions on new data. 

```Python
prediction = model.predict(file="unseen_data.csv")
```

All required features must be present in the `unseen_data.csv` file. It is possible for the model to now use all features provided to the `train` function. In case a partial feature list is chosen, only the selected features are required to be provided. 

# Feature Selection
Framework automatically performs a feature selection. All features (except target) are considered by default for feature selection.
Framework is smart enough to remove ID / Primary key columns. 

You can manually specify a subset of features to be used for training. An automatic feature selection will still be carried out, but will be restricted to the subset of features provided. 

``` Python
model = bc.train(file="data.csv", target="Y_value", features=["col1", "col2", "col3"])
```

This does not guarantee that all specified features will be used in the final model. The framework will perform an automated feature selection from amongst these features. This only guarantees that other features if present in the data will not be considered. 

```Python
model.features() #prints the features selected by the model

model.plot_feature_importance() #shows a feature importance graph
```

# Printing Model Stats
``` Python
model.stats()
```

Print the key model parameters, such as Precision, Recall, F1-Score. The parameters change based on the type of AutoAI problem. 

# Persistence
``` Python
model.save('./my_model.pkl')

model = bc.load('./my_model.pkl')
```

You can save a trained model, and load it in the future to generate predictions. 

# Accelerated Training
Leverage BlobCity AI Cloud for fast training on large datasets. Reasonable cloud infrastructure included for free.

[![BlobCity AI Cloud](https://shields.io/badge/Run%20On-BlobCity-orange)](https://pix.blobcity.com/pgMuJMLv)
[![CPU](https://shields.io/badge/CPU-Free-blue)](https://pix.blobcity.com/pgMuJMLv)
[![GPU](https://shields.io/badge/GPU-%2475%2Fmonth-green)](https://pix.blobcity.com/pgMuJMLv)


# Features and Roadmap
- [x] Numercial data Classification and Regression
- [x] Automatic feature selection
- [x] Code generation
- [ ] Neural Networks & Deep Learning
- [ ] Image classification
- [ ] Optical Character Recognition (english only)
- [ ] Video tagging with YOLO
- [ ] Generative AI using GAN
