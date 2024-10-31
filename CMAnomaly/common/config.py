import os
import time
import argparse
import logging
import glob

subdatasets = {
    "industry": [
        "e29ca1cd",
        "c23b2b2d",
        "aeb5a1de",
        "2fe95315",
        "0a82a873",
        "af732cc4",
        "b2a04b7f",
        "c2970798",
        "5dafb960"
    ],
}


def parse_multi_setting(setting):
    result_dict = {}
    if setting:
        for item in setting:
            k, v = item.strip().split("=")
            try:
                result_dict[k] = eval(v)
            except NameError:
                result_dict[k] = str(v)
    return result_dict


def parse_arguments():
    import torch

    parser = argparse.ArgumentParser(
        description="Anomaly detection repository for TS datasets"
    )
    parser.add_argument(
        "--subdataset",
        type=str,
        metavar="D",
        default="",
        help="dataset name",
    )

    parser.add_argument(
        "--dataset", type=str, metavar="D", default="SMD", help="dataset name"
    )

    # SMD: "./datasets/anomaly/SMD/processed"
    # SMAP: "./datasets/anomaly/SMAP-MSL/processed_SMAP"
    # MSL: "./datasets/anomaly/SMAP-MSL/processed_MSL"
    # Simulated: "./datasets/Simulated/simulated_p0.1.csv"
    parser.add_argument(
        "--path",
        type=str,
        metavar="PATH",
        # default="./datasets/anomaly/SMAP-MSL/",
        default="./datasets/anomaly/SMAP-MSL/processed_SMAP",
        help="path where the dataset is cmanomalyd",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        metavar="PATH",
        default="./checkpoints",
        help="path where the estimator is/should be saved",
    )

    parser.add_argument("--set", metavar="KEY=VALUE", nargs="+")

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        metavar="GPU",
        help="index of GPU used for computations (default: 0)",
    )
    parser.add_argument(
        "--expid", type=str, default="mlstm_250", help="Expid in hypers"
    )
    parser.add_argument(
        "--load",
        type=str,
        default="",
        help="Load a pretrained model from the specific directory",
    )
    parser.add_argument(
        "--nrows", type=int, default=None, help="Read only first nrows for test"
    )
    parser.add_argument(
        "--clear",
        type=int,
        default=None,
        help="Set to 1 if data re-processsing is needed",
    )

    args = parser.parse_args()
    if args.gpu >= 0 and torch.cuda.is_available():
        args.device = torch.device("cuda:{}".format(args.gpu))
    else:
        args.device = torch.device("cpu")
    os.makedirs(args.save_path, exist_ok=True)
    return vars(args)


def initialize_config(config_dir, args):
    params = dict()
    model_configs = glob.glob(os.path.join(config_dir, "*/*.yaml")) + glob.glob(
        os.path.join(config_dir, "*.yaml")
    )

    if not model_configs:
        raise RuntimeError("config_dir={} is not valid!".format(config_dir))
    found_params = find_config(model_configs, args["expid"])
    base_config = found_params.get("Base", {})
    model_config = found_params.get(args["expid"])
    params.update(base_config)
    params.update(args)
    params.update(model_config)

    params = set_logger(params)
    params.update(parse_multi_setting(args["set"]))

    with open(os.path.join(params["save_path"], "model_config.yaml"), "w") as fr:
        found_params["Base"]["save_path"] = params["save_path"]
        found_params["Base"]["trial_id"] = params["trial_id"]
        yaml.dump(found_params, fr)

    if params["dataset"] in ["SMAP", "MSL"]:
        params["prediction_dims"] = [0]
    else:
        params["prediction_dims"] = []

    return params


def get_trial_id():
    trial_id = nni.get_trial_id()
    if trial_id == "STANDALONE":
        trial_id = time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time()))
    return trial_id


def set_logger(params):
    if not params["load"]:
        trial_id = get_trial_id()
        log_dir = os.path.join(params["save_path"], trial_id)
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = params["save_path"]
        trial_id = params["trial_id"]
    log_file = os.path.join(log_dir, "{}.log".format(trial_id))

    # logs will not show in the file without the two lines.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    # update save_path
    params["save_path"] = log_dir
    params["trial_id"] = trial_id
    return params


def find_config(model_configs, experiment_id):
    found_params = dict()
    for config in model_configs:
        with open(config, "r", encoding="utf-8") as cfg:
            config_dict = yaml.safe_load(cfg)
            if "Base" in config_dict:
                found_params["Base"] = config_dict["Base"]
            if experiment_id in config_dict:
                found_params[experiment_id] = config_dict[experiment_id]
        if len(found_params) == 2:
            break
    return found_params
