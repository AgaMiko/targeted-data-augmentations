import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import os
import numpy as np
import statistics
import csv
from sklearn.metrics import f1_score, recall_score, precision_score
import glob
import argparse
import timm
import yaml
from datasets.transforms_aug import get_transforms, get_augmentation
from models.model import LesionClassification


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_config_path",
        type=str,
        default='config.yml',
        help="path to .yaml config file specifying hyperparameters of different model sections."
    )
    return parser


def make_prediction(batch, model):
    """
    Calculates predictions for given batch of in puts.
    """
    images, labels = batch
    outputs = torch.softmax(model(images),dim=0)
    predicted = torch.argmax(outputs,dim=1).cpu().detach().numpy()
    prob = dict()
    prob["ben"] = outputs.reshape(2,-1)[0].cpu().detach().numpy()
    prob["mal"] = outputs.reshape(2,-1)[1].cpu().detach().numpy()
    return list(predicted), list(labels.cpu().detach().numpy()), prob


def switched_score(pred_labels, aug_labels):
    """
    Calculates number of switched predictions.

    This function gets predicted labels for image without any inserted bias,
    and compares it with prediction of the same input but with inserted bias.
    Number of predictions that changed (switched) class is counted.
    """
    switched = 0
    ben_to_mal = 0
    mal_to_ben = 0
    for pred, aug in zip(pred_labels, aug_labels):
        if pred != aug:
            switched += 1
            if pred == "benign" or pred == 0 or pred == "0":
                ben_to_mal += 1
            elif pred == "malignant" or pred == 1 or pred == "1":
                mal_to_ben += 1
    print("Switched classes", switched, "out of", len(
        pred_labels), "--", switched/len(pred_labels)*100, "%")
    print("Switched benign to malignant", ben_to_mal)
    print("Switched malignant to benign", mal_to_ben)
    return switched, ben_to_mal, mal_to_ben


def update_probs(probs, prob_clean, prob_aug):
    for key in ["mal", "ben"]:
        probs["clean"][key].append(prob_clean[key])
        probs["aug"][key].append(prob_clean[key])
    return probs


def calculate_difference(probs):
    diff = {
        "mal": list(),
        "ben": list()
    }
    for key in ["mal", "ben"]:
        for clean, biased in zip(probs["clean"][key], probs["aug"][key]):
            diff[key].append(abs(clean - biased))
    return diff


def main(args):
    with open(args.model_config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    header = ['model', 'aug_type',
              'switched', 'mal_to_ben', 'ben_to_mal',
              'f1', 'f1_aug', 'recall', 'recall_aug',
              'precision', 'precision_aug',
              'mean_diff_ben', 'mean_diff_mal']
    models_to_explain = glob.glob(os.path.join(
        config["cbi_plan"]["dir_to_saved_models"], "*.ckpt"))
    with open(config["cbi_plan"]["results_path"], 'w', encoding='UTF-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    # for each model from models_to_explain run Counterfactual Experiments
    # run aug_list for each model
    # each aug will be run with different settings
    augmentation_names = config["cbi_plan"]["augmentations"]
    for model_path in models_to_explain:
        filename = model_path.split("/")[-1]
        for arch_nr, iter_model_name in enumerate(config["cbi_plan"]["architectures"]["names"]):
            if iter_model_name in filename:
                model_name = iter_model_name
                img_size = config["cbi_plan"]["architectures"]["input_sizes"][arch_nr]
                batch_size = config["cbi_plan"]["architectures"]["batch_sizes"][arch_nr]
                file_temp = filename.split(
                    iter_model_name)[-1].split(".ckpt")[0]
                aug_prob = file_temp.split("_")[-1]
                aug_train = file_temp.split(aug_prob)[0]

                # load trained classification model
                model = LesionClassification(model=model_name,
                                            num_classes=2,
                                            mode="cbi")
                model.load_from_checkpoint(model_path,
                                        model=model_name,
                                        num_classes=2,
                                        mode="cbi")
                model.eval()
                model = model.to("cuda")
                # counterfactual experiments start
                for aug_nr, aug_type in enumerate(augmentation_names):
                    mask_dir = config["mask_dirs"]["test"][aug_type]

                    # biased data
                    biased_test_transform, _ = get_transforms(
                        img_size, type_aug=aug_type, mask_dir=mask_dir, aug_p=1.0)
                    biased_test_transform = get_augmentation(biased_test_transform)
                    biased_test_set = datasets.ImageFolder(root=config["general"]["test_dir"],
                                                        transform=biased_test_transform)
                    biased_test_loader = DataLoader(
                        biased_test_set, batch_size=batch_size, shuffle=False, num_workers=4)
                    classes = list(biased_test_set.class_to_idx.keys())

                    # clean data
                    _, clean_test_transform = get_transforms(
                        img_size, type_aug="normal", mask_dir=mask_dir, aug_p=0.0)
                    clean_test_transform = get_augmentation(clean_test_transform)
                    clean_test_set = datasets.ImageFolder(root=config["general"]["test_dir"],
                                                        transform=clean_test_transform)
                    clean_test_loader = DataLoader(
                        clean_test_set, batch_size=batch_size, shuffle=False, num_workers=4)

                    prob_diff = list()
                    org_labels = list()
                    pred_labels = list()
                    aug_labels = list()
                    preds = list()
                    preds_aug = list()
                    probs = dict()
                    probs["clean"] = {
                        "mal": list(),
                        "ben": list(),
                    }
                    probs["aug"] = {
                        "mal": list(),
                        "ben": list(),
                    }

                    for i, (batch, batch_aug) in tqdm(enumerate(zip(clean_test_loader, biased_test_loader))):
                        predicted_class, labels, prob_clean = make_prediction(
                            batch, model)
                        predicted_class_aug, labels, prob_aug = make_prediction(
                            batch_aug, model)

                        preds.extend(predicted_class)
                        preds_aug.extend(predicted_class_aug)
                        probs = update_probs(probs, prob_clean, prob_aug)

                        org_labels.extend(labels)
                        # prob_diff.append(np.abs(probability-probability_aug))

                        pred_labels.extend(predicted_class)
                        aug_labels.extend(predicted_class_aug)

                    print("===============SUMMARY===============")
                    print(model_name)
                    f1 = f1_score(org_labels, preds, zero_division=1)
                    f1_aug = f1_score(org_labels, preds_aug, zero_division=1)
                    recall = recall_score(org_labels, preds, zero_division=1)
                    recall_aug = recall_score(
                        org_labels, preds_aug, zero_division=1)
                    precision = precision_score(org_labels, preds, zero_division=1)
                    precision_aug = precision_score(
                        org_labels, preds_aug, zero_division=1)
                    switched, ben_to_mal, mal_to_ben = switched_score(
                        pred_labels, aug_labels)
                    diff = calculate_difference(probs)
                    mean_diff_ben = statistics.mean(diff["ben"])
                    mean_diff_mal = statistics.mean(diff["mal"])
                    # counterfactual experiments end
                    print("Saving results to file...")
                    data = [model_name, augmentation_names[aug_nr],
                            switched, mal_to_ben, ben_to_mal, 
                            f1, f1_aug, recall, recall_aug, precision, precision_aug,
                            mean_diff_ben, mean_diff_mal]

                    with open(config["cbi_plan"]["results_path"], 'a', encoding='UTF-8', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(data)

                    data = [
                        probs["clean"]["mal"],
                        probs["clean"]["ben"],
                        probs["aug"]["mal"],
                        probs["aug"]["ben"],
                        labels,
                    ]
                    header = ["prob_clean_mal", "prob_clean_ben",
                            "prob_bias_mal", "prob_bias_ben", "label"]
                    
                    new_file = config["cbi_plan"]["save_preds_dir"] + filename + ".csv"
                    with open(new_file, 'w', encoding='UTF-8', newline='') as f:
                    
                        writer = csv.writer(f)
                        writer.writerow(header)
                        writer.writerow(data)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
