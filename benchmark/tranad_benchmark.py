import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys

sys.path.append("../")
import logging
from networks.tranad import *
from common import data_preprocess
from common.dataloader import load_dataset, get_dataloaders
from common.utils import seed_everything, load_config, set_logger, print_to_json
from networks.tranad.models import TranAD
from common.evaluation import Evaluator, TimeTracker
from common.exp import store_entity

seed_everything()
if __name__ == "__main__":
    import argparse
    for i in range(0,5):
        datastrlist=["tranad_SMD","tranad_PSM","tranad_SWAT","tranad_MSL","tranad_SMAP"]
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--config",
            type=str,
            default="./benchmark_config/",
            help="The config directory.",
        )
        parser.add_argument("--expid", type=str, default="tranad_SMD")
        parser.add_argument("--gpu", type=int, default=-1)
        args = vars(parser.parse_args())

        config_dir = args["config"]
        experiment_id = args["expid"]

        params = load_config(config_dir, experiment_id)
        set_logger(params, args)
        logging.info(print_to_json(params))

        data_dict = load_dataset(
            data_root=params["data_root"],
            entities=params["entities"],
            valid_ratio=params["valid_ratio"],
            dim=params["dim"],
            test_label_postfix=params["test_label_postfix"],
            test_postfix=params["test_postfix"],
            train_postfix=params["train_postfix"],
            nrows=params["nrows"],
        )

        # preprocessing
        pp = data_preprocess.preprocessor(model_root=params["model_root"])
        data_dict = pp.normalize(data_dict, method=params["normalize"])

        # sliding windows
        window_dict = data_preprocess.generate_windows(
            data_dict,
            window_size=params["window_size"],
            stride=params["stride"],
        )

        # train/test on each entity put here
        evaluator = Evaluator(**params["eval"])
        for entity in params["entities"]:
            logging.info("Fitting dataset: {}".format(entity))
            windows = window_dict[entity]
            train_windows = windows["train_windows"]
            test_windows = windows["test_windows"]

            train_loader, _, test_loader = get_dataloaders(train_windows, test_windows)

            model = TranAD(
                params["dim"],
                params["window_size"],
                lr=params["lr"],
                model_root=params["model_root"],
                device=params["device"],
            )

            tt = TimeTracker(nb_epoch=params["nb_epoch"])

            tt.train_start()
            model.fit(
                params["nb_epoch"],
                train_loader,
                training=True,
            )
            tt.train_end()

            train_anomaly_score = model.predict_prob(train_loader)

            tt.test_start()
            anomaly_score, anomaly_label = model.predict_prob(
                test_loader, windows["test_label"]
            )
            tt.test_end()

            store_entity(
                params,
                entity,
                train_anomaly_score,
                anomaly_score,
                anomaly_label,
                time_tracker=tt.get_data(),
            )
        evaluator.eval_exp(
            exp_folder=params["model_root"],
            entities=params["entities"],
            merge_folder=params["benchmark_dir"],
            extra_params=params,
        )
