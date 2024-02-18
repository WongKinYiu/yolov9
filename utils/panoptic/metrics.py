import numpy as np
import torch

from ..metrics import ap_per_class


def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.1, 0.9, 0.1, 0.9]
    return (x[:, :len(w)] * w).sum(1)


def ap_per_class_box_and_mask(
        tp_m,
        tp_b,
        conf,
        pred_cls,
        target_cls,
        plot=False,
        save_dir=".",
        names=(),
):
    """
    Args:
        tp_b: tp of boxes.
        tp_m: tp of masks.
        other arguments see `func: ap_per_class`.
    """
    results_boxes = ap_per_class(tp_b,
                                 conf,
                                 pred_cls,
                                 target_cls,
                                 plot=plot,
                                 save_dir=save_dir,
                                 names=names,
                                 prefix="Box")[2:]
    results_masks = ap_per_class(tp_m,
                                 conf,
                                 pred_cls,
                                 target_cls,
                                 plot=plot,
                                 save_dir=save_dir,
                                 names=names,
                                 prefix="Mask")[2:]

    results = {
        "boxes": {
            "p": results_boxes[0],
            "r": results_boxes[1],
            "ap": results_boxes[3],
            "f1": results_boxes[2],
            "ap_class": results_boxes[4]},
        "masks": {
            "p": results_masks[0],
            "r": results_masks[1],
            "ap": results_masks[3],
            "f1": results_masks[2],
            "ap_class": results_masks[4]}}
    return results


class Metric:

    def __init__(self) -> None:
        self.p = []  # (nc, )
        self.r = []  # (nc, )
        self.f1 = []  # (nc, )
        self.all_ap = []  # (nc, 10)
        self.ap_class_index = []  # (nc, )

    @property
    def ap50(self):
        """AP@0.5 of all classes.
        Return:
            (nc, ) or [].
        """
        return self.all_ap[:, 0] if len(self.all_ap) else []

    @property
    def ap(self):
        """AP@0.5:0.95
        Return:
            (nc, ) or [].
        """
        return self.all_ap.mean(1) if len(self.all_ap) else []

    @property
    def mp(self):
        """mean precision of all classes.
        Return:
            float.
        """
        return self.p.mean() if len(self.p) else 0.0

    @property
    def mr(self):
        """mean recall of all classes.
        Return:
            float.
        """
        return self.r.mean() if len(self.r) else 0.0

    @property
    def map50(self):
        """Mean AP@0.5 of all classes.
        Return:
            float.
        """
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map(self):
        """Mean AP@0.5:0.95 of all classes.
        Return:
            float.
        """
        return self.all_ap.mean() if len(self.all_ap) else 0.0

    def mean_results(self):
        """Mean of results, return mp, mr, map50, map"""
        return (self.mp, self.mr, self.map50, self.map)

    def class_result(self, i):
        """class-aware result, return p[i], r[i], ap50[i], ap[i]"""
        return (self.p[i], self.r[i], self.ap50[i], self.ap[i])

    def get_maps(self, nc):
        maps = np.zeros(nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps

    def update(self, results):
        """
        Args:
            results: tuple(p, r, ap, f1, ap_class)
        """
        p, r, all_ap, f1, ap_class_index = results
        self.p = p
        self.r = r
        self.all_ap = all_ap
        self.f1 = f1
        self.ap_class_index = ap_class_index


class Metrics:
    """Metric for boxes and masks."""

    def __init__(self) -> None:
        self.metric_box = Metric()
        self.metric_mask = Metric()

    def update(self, results):
        """
        Args:
            results: Dict{'boxes': Dict{}, 'masks': Dict{}}
        """
        self.metric_box.update(list(results["boxes"].values()))
        self.metric_mask.update(list(results["masks"].values()))

    def mean_results(self):
        return self.metric_box.mean_results() + self.metric_mask.mean_results()

    def class_result(self, i):
        return self.metric_box.class_result(i) + self.metric_mask.class_result(i)

    def get_maps(self, nc):
        return self.metric_box.get_maps(nc) + self.metric_mask.get_maps(nc)

    @property
    def ap_class_index(self):
        # boxes and masks have the same ap_class_index
        return self.metric_box.ap_class_index


class Semantic_Metrics:
    def __init__(self, nc, device):
        self.nc = nc  # number of classes
        self.device = device
        self.iou = []
        self.c_bit_counts = torch.zeros(nc, dtype = torch.long).to(device)
        self.c_intersection_counts = torch.zeros(nc, dtype = torch.long).to(device)
        self.c_union_counts = torch.zeros(nc, dtype = torch.long).to(device)

    def update(self, pred_masks, target_masks):
        nb, nc, h, w = pred_masks.shape
        device = pred_masks.device

        for b in range(nb):
            onehot_mask = pred_masks[b].to(device)
            # convert predict mask to one hot
            semantic_mask = torch.flatten(onehot_mask, start_dim = 1).permute(1, 0) # class x h x w -> (h x w) x class
            max_idx = semantic_mask.argmax(1)
            output_masks = (torch.zeros(semantic_mask.shape).to(self.device)).scatter(1, max_idx.unsqueeze(1), 1.0) # one hot: (h x w) x class
            output_masks = torch.reshape(output_masks.permute(1, 0), (nc, h, w)) # (h x w) x class -> class x h x w
            onehot_mask = output_masks.int()

            for c in range(self.nc):
                pred_mask = onehot_mask[c].to(device)
                target_mask = target_masks[b, c].to(device)

                # calculate IoU
                intersection = (torch.logical_and(pred_mask, target_mask).sum()).item()
                union = (torch.logical_or(pred_mask, target_mask).sum()).item()
                iou = 0. if (0 == union) else (intersection / union)

                # record class pixel counts, intersection counts, union counts
                self.c_bit_counts[c] += target_mask.int().sum()
                self.c_intersection_counts[c] += intersection
                self.c_union_counts[c] += union

                self.iou.append(iou)

    def results(self):
        # Mean IoU
        miou = 0. if (0 == len(self.iou)) else np.sum(self.iou) / (len(self.iou) * self.nc)

        # Frequency Weighted IoU
        c_iou = self.c_intersection_counts / (self.c_union_counts + 1)  # add smooth
        # c_bit_counts = self.c_bit_counts.astype(int)
        total_c_bit_counts = self.c_bit_counts.sum()
        freq_ious = torch.zeros(1, dtype = torch.long).to(self.device) if (0 == total_c_bit_counts) else (self.c_bit_counts / total_c_bit_counts) * c_iou
        fwiou = (freq_ious.sum()).item()

        return (miou, fwiou)

    def reset(self):
        self.iou = []
        self.c_bit_counts = torch.zeros(self.nc, dtype = torch.long).to(self.device)
        self.c_intersection_counts = torch.zeros(self.nc, dtype = torch.long).to(self.device)
        self.c_union_counts = torch.zeros(self.nc, dtype = torch.long).to(self.device)


KEYS = [
    "train/box_loss",
    "train/seg_loss",  # train loss
    "train/cls_loss",
    "train/dfl_loss",
    "train/fcl_loss",
    "train/dic_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP_0.5(B)",
    "metrics/mAP_0.5:0.95(B)",  # metrics
    "metrics/precision(M)",
    "metrics/recall(M)",
    "metrics/mAP_0.5(M)",
    "metrics/mAP_0.5:0.95(M)",  # metrics
    "metrics/MIOUS(S)",
    "metrics/FWIOUS(S)",        # metrics
    "val/box_loss",
    "val/seg_loss",  # val loss
    "val/cls_loss",
    "val/dfl_loss",
    "val/fcl_loss",
    "val/dic_loss",
    "x/lr0",
    "x/lr1",
    "x/lr2",]

BEST_KEYS = [
    "best/epoch",
    "best/precision(B)",
    "best/recall(B)",
    "best/mAP_0.5(B)",
    "best/mAP_0.5:0.95(B)",
    "best/precision(M)",
    "best/recall(M)",
    "best/mAP_0.5(M)",
    "best/mAP_0.5:0.95(M)",
    "best/MIOUS(S)",
    "best/FWIOUS(S)",]
