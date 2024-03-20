import os
from utils.general import LOGGER, colorstr
from pathlib import Path

try:
    import mlflow
except (ModuleNotFoundError, ImportError):
    mlflow = None

PREFIX = colorstr("MLflow: ")
SANITIZE = lambda x: {k.replace(':', '_').replace(';', '_'): v for k, v in x.items()}


class MLflowLogger:
    """Log metrics, parameters, models and much more with MLflow"""


    def __init__(self, opt, hyp) -> None:

        self.mlflow = mlflow
        self.data_dict = None
        self.opt = opt
        self.hyp = hyp


    def on_pretrain_routine_end(self, paths):

        global mlflow

        uri = os.environ.get("MLFLOW_TRACKING_URI")
        LOGGER.info(f"{PREFIX} tracking uri: {uri}")
        mlflow.set_tracking_uri(uri)
        experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME") or self.opt.save_dir.split('/')[-2]
        run_name = os.environ.get("MLFLOW_RUN") or self.opt.save_dir.split('/')[-1]
        mlflow.set_experiment(experiment_name)
        mlflow.pytorch.autolog()

        try:
            active_run = mlflow.active_run() or mlflow.start_run(run_name=run_name)
            LOGGER.info(f"{PREFIX}logging run_id({active_run.info.run_id}) to {uri}")
            
            # Don't save hyps from opt directly
            mlflow.log_params({f"param/{key}": value for key, value in vars(self.opt).items() if key != 'hyp'})
            mlflow.log_params({f"hyp/{key}": value for key, value in self.hyp.items()})

        except Exception as e:
            LOGGER.warning(f"{PREFIX}WARNING ⚠️ Failed to initialize: {e}\n" f"{PREFIX}WARNING ⚠️ Not tracking this run")
            self.mlflow = None


        for path in paths:
            mlflow.log_artifact(str(path))

    
    def on_fit_epoch_end(self, vals, epoch):
        """Log training metrics at the end of each fit epoch to MLflow."""
        
        if self.mlflow:
            mlflow.log_metrics(metrics=SANITIZE(vals), step=epoch)


    def on_model_save(self, last):
        if self.mlflow:
            mlflow.log_artifacts(os.path.dirname(last), artifact_path="weights")


    def on_train_end(self, save_dir):
        """Log model artifacts at the end of the training."""

        if self.mlflow:
            mlflow.log_artifacts(save_dir)

            mlflow.end_run()
            
            LOGGER.info(
                f"{PREFIX}results logged to {mlflow.get_tracking_uri()}\n"
            )
        
        
