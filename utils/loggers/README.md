# Loggers

## Supported platforms

* [TensorBoard](https://www.tensorflow.org/tensorboard)
* [Weights & Biases](https://wandb.ai/)
* [ClearML](https://clear.ml/)
* [Comet](https://www.comet.com/)
* [MLflow](https://mlflow.org/)

## How to use ?

### TensorBoard

TODO

### Weights & Biases

TODO

### ClearML

TODO

### Comet

TODO

### MLflow

MLflow is an open-source platform for monitoring and managing machine learning experiments.

1. __Prerequisites__

Make sure you have installed the MLflow library:

    pip install mlflow

2. __Serveur MLflow__

Launch your MLflow server. You can run it with the following command:

    mlflow server --backend-store-uri mlflow_server

This will start a local server at http://127.0.0.1:5000 by default and save all mlflow logs to the `mlflow_server` directory at the location of the command execution.

To cut all instances of MLflow, you can run this command:

    ps aux | grep 'mlflow' | grep -v 'grep' | awk '{print $2}' | xargs kill -9

3. __MLflow parameters__

Set your server address in the `MLFLOW_TRACKING_URI` environment variable. If the address is not provided, a warning will be raised and the run will not be recorded.

Set the name of your experiment in the `MLFLOW_EXPERIMENT_NAME` environment variable. If no name is provided, the project name (--project of train.py) will be set by default.

Define the name of your run in the `MLFLOW_RUN` environment variable. If no name is provided, the run name (--name of train.py) will be set by default.

After that, your training sessions will be saved in your MLflow server!
