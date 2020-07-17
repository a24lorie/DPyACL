***
                                 DPyACL
                Distributed Python Framework for Active Learning

                               May 2020
			    Alfredo Lorie Bernardo							

                             version 0.3.1

***

# Introduction

`DPyACL` is a flexible Distributed Active Learning library written in Python, aimed to make active learning experiments 
simpler and faster. Its leverage Dask distributed features to execute active learning experiments computations among a 
cluster of computers, allowing to speed up computation and tackle scenarios where data doesn't fit in a single computer. 
It also has been developed with a modular object-oriented design to provide an intuitive, ease of use interface and 
to allow reuse, modification, and extensibility. It also offers full compatibility with libraries like NumPy, SciPy, 
Pandas, Scikit-learn and  Keras. This library is available in PyPI and distributed under the GNU license.4 


Up to date, DPyACL heavily uses Dask library to implement in a distributed and parallel fashion the the most significant 
strategies strategies that have appeared on the single_label-label. 
For next versions, we hope to include strategies strategies related with  multi-label learning paradigms.

# Download

GitHub: <https://github.com/a24lorie/DPyACL>

# Using DPyACL

The fastest way to use `DPyACL` is from a Jupyter Notebook. 

## Preparing an experiment

When defining an Active Learning experiment `DPyACL` offers set pre-defined components that can be configured and 
combined by the user to better fit its needs. The required components to setup and experiment are listed below    

 1. **The Dataset**
 2. **Labelled and unlabelled sets**: Optional - The experiment might be configured to randomly choose an initial labeled and unlabeled sets
 2. **An Experiment**: HoldOut and KFold experiments are provided
 3. **The AL scenario**: The current release provides a Pool Based Scenario 
 4. **The Machine Learning Technique**: It can be a machine learning technique from any library that provides an API compatible with the 
            **fit**, **predict** and **predict_proba** definitions. Sklearn, Dask-ML, Keras are compatible 
 5. **The Evaluation Method(s)**
 7. **The Query Strategy**
 5. **The Stopping Criteria**
 8. **The Oracle**: The current release provides a Simulated Oracle
 
 
### Configuring the experiment
 
```python
ml_technique = LogisticRegression(solver='liblinear')
stopping_criteria = MaxIteration(50)
query_strategy =  QueryMarginSampling()
performance_metrics = [
                Accuracy(),
                F1(average='macro'),
                Precision(average='macro'),
                Recall(average='macro')]

experiment = HoldOutExperiment(
    _X,
    _y,
    scenario_type=PoolBasedSamplingScenario,
    train_idx=train_idx,
    test_idx=test_idx,
    label_idx=label_idx,
    unlabel_idx=unlabel_idx,
    ml_technique=ml_technique,
    performance_metrics=performance_metrics,
    query_strategy=query_strategy,
    oracle=SimulatedOracleQueryIndex(labels=_y),
    stopping_criteria=stopping_criteria,
    self_partition=False
)
```

### Execute the experiment
 
```python
 result = experiment.evaluate(verbose=True)
```

### Analyze the experiment results
```python
 query_analyser = ExperimentAnalyserFactory.experiment_analyser(
                            performance_metrics= [metric.metric_name for metric in performance_metrics],
                            method_name=query_strategy.query_function_name,
                            method_results=result,
                            type="queries"
                        )

# get a brief description of the experiment
query_analyser.plot_learning_curves(title='Active Learning experiment results')
```

# Contribution

If you find a bug, send a [pull request](https://github.com/a24lorie/PyACL/pulls) and we'll discuss things. If you are not familiar with "***pull request***" term I recommend reading the following [article](https://yangsu.github.io/pull-request-tutorial/) for better understanding   
