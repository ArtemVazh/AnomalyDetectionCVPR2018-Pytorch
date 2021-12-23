import argparse
import os
import torch
import torch.backends.cudnn as cudnn

from network.TorchUtils import TorchModel
from features_loader import FeaturesLoaderVal, FeaturesLoader, FeaturesLoaderTrain
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from os import path
import numpy as np
from scipy import spatial


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Video Classification Parser")
    parser.add_argument(
        "--features_path", default="./data/anomaly_features", help="path to features"
    )
    parser.add_argument(
        "--annotation_path", default="Test_Annotation.txt", help="path to annotations"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./exps/model.weights",
        help="set logging file.",
    )
    parser.add_argument(
        "--calc_mode", type=str, default="mil", help="metrics calculation mode"
    )

    parser.add_argument(
        "--train_features_path",
        default="data/anomaly_features",
        help="path to features",
    )
    parser.add_argument(
        "--train_annotation_path",
        default="Train_Annotation.txt",
        help="path to annotations",
    )
    parser.add_argument(
        "--filename", type=str, default="roc_auc.png", help="filename for ROC-AUC image"
    )

    save_to_pt_folder_parser = parser.add_mutually_exclusive_group(required=False)
    save_to_pt_folder_parser.add_argument(
        "--save_to_pt_folder", dest="save_to_pt_folder", action="store_true"
    )
    save_to_pt_folder_parser.add_argument(
        "--no_save_to_pt_folder", dest="save_to_pt_folder", action="store_false"
    )
    save_to_pt_folder_parser.set_defaults(save_to_pt_folder=False)

    use_centroid_parser = parser.add_mutually_exclusive_group(required=False)
    use_centroid_parser.add_argument(
        "--use_centroid", dest="use_centroid", action="store_true"
    )
    use_centroid_parser.add_argument(
        "--no_use_centroid", dest="use_centroid", action="store_false"
    )
    use_centroid_parser.set_defaults(use_centroid=False)

    return parser.parse_args()


def get_centroids(model, data_iter):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normal_embed = None
    anomaly_embed = None

    n_normal = 0
    n_anomaly = 0

    with torch.no_grad():
        for features, labels in tqdm(data_iter):
            # features is a batch where each item is a tensor of 32 4096D features
            features = features.to(device)
            labels = labels  # .squeeze()

            outputs = model(features)  # (batch_size, 32, embed_size)

            embed = outputs.mean(1).cpu().numpy()
            y_true = labels.cpu().numpy()

            if (y_true == 0).sum():
                if normal_embed is None and (y_true == 0).sum():
                    normal_embed = embed[y_true == 0].mean(0)
                else:
                    normal_embed += embed[y_true == 0].mean(0)

            if (y_true == 1).sum():
                if anomaly_embed is None:
                    anomaly_embed = embed[y_true == 1].mean(0)
                else:
                    anomaly_embed += embed[y_true == 1].mean(0)

            n_normal += (y_true == 0).sum()
            n_anomaly += (y_true == 1).sum()

            torch.cuda.empty_cache()

    centroids = np.array([normal_embed / n_normal, anomaly_embed / n_anomaly])
    return centroids


if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader = FeaturesLoaderVal(
        features_path=args.features_path, annotation_path=args.annotation_path
    )

    data_iter = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # 4, # change this part accordingly
        pin_memory=True,
    )

    train_data_loader = FeaturesLoaderTrain(
        features_path=args.train_features_path,
        annotation_path=args.train_annotation_path,
    )

    train_data_iter = torch.utils.data.DataLoader(
        train_data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # 4, # change this part accordingly
        pin_memory=True,
    )

    model = TorchModel.load_model(args.model_path).to(device).eval()

    # enable cudnn tune
    cudnn.benchmark = True

    y_trues = torch.tensor([])
    y_preds = torch.tensor([])

    if args.use_centroid:
        print("Use normal centroids to calculate the anomaly score")
        centroids = get_centroids(model, train_data_iter)
    else:
        print("Use 1st video segment to calculate the anomaly score")

    k = 0
    with torch.no_grad():
        for features, start_end_couples, lengths in tqdm(data_iter):
            # features is a batch where each item is a tensor of 32 4096D features
            features = features.to(device)
            if args.calc_mode == "mil":
                outputs = model(features).squeeze(-1)  # (batch_size, 32)
            else:  # in case of Triplet loss
                outputs = model(features)  # (batch_size, 32, embed_size)
            for vid_len, couples, output in zip(
                lengths, start_end_couples, outputs.cpu().numpy()
            ):
                y_true = np.zeros(vid_len)
                y_pred = np.zeros(vid_len)

                segments_len = vid_len // 32
                for couple in couples:
                    if couple[0] != -1:
                        y_true[couple[0] : couple[1]] = 1

                for i in range(32):
                    segment_start_frame = i * segments_len
                    segment_end_frame = (i + 1) * segments_len
                    if args.calc_mode == "triplet":
                        if args.use_centroid:
                            y_pred[segment_start_frame:segment_end_frame] = (
                                (output[i] - centroids[0]) ** 2
                            ).sum(-1)
                        else:
                            y_pred[segment_start_frame:segment_end_frame] = (
                                (output[i] - output[0]) ** 2
                            ).sum(-1)
                    else:  # default MIL calculation
                        y_pred[segment_start_frame:segment_end_frame] = output[i]

                if y_trues is None:
                    y_trues = y_true
                    y_preds = y_pred
                else:
                    y_trues = np.concatenate([y_trues, y_true])
                    y_preds = np.concatenate([y_preds, y_pred])
                    
            #path = 'exps/grid_search/triplet_adadelta_0.1_8e-05_5_5_0.2/42/models/vectors'
            #with open(f'{path}/{k}_embed.npy', 'wb') as f:
            #    np.save(f, outputs.detach().cpu().numpy()[0, :, :])
            #with open(f'{path}/{k}_true.npy', 'wb') as f:
            #    np.save(f, y_true)
            #with open(f'{path}/{k}_pred.npy', 'wb') as f:
            #    np.save(f, y_pred)
            k+=1

    fpr, tpr, thresholds = roc_curve(y_true=y_trues, y_score=y_preds, pos_label=1)

    plt.figure()
    lw = 2
    roc_auc = auc(fpr, tpr)
    plt.plot(
        fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")

    os.makedirs("graphs", exist_ok=True)
    plt.savefig(f"graphs/{args.filename}")
    print("ROC curve (area = %0.3f)" % roc_auc)
    if args.save_to_pt_folder:
        save_folder_path = f"{os.sep}".join(args.model_path.split(os.sep)[:-1])
        save_name = args.model_path.split(os.sep)[-1].split(".")[0]
        plt.savefig(path.join(save_folder_path, save_name + "_roc_auc.png"))
        with open(path.join(save_folder_path, save_name + "_roc_auc.txt"), "w") as f:
            f.write("ROC curve (area = %0.2f)" % roc_auc)
    plt.close()
