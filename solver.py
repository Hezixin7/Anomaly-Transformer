import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment
from sklearn.metrics import roc_auc_score


def extract_segments(gt_01):
    segs = []
    in_seg = False
    s = 0
    for i, v in enumerate(gt_01):
        if v == 1 and not in_seg:
            in_seg = True
            s = i
        elif v == 0 and in_seg:
            in_seg = False
            segs.append((s, i - 1))
    if in_seg:
        segs.append((s, len(gt_01) - 1))
    return segs


def early_detection_metrics(scores, gt, thresh, ks=(5, 10, 20)):
    """
    scores, gt: 1D numpy arrays length T_total (point-wise)
    strict early horizon: [s-k, s-1]
    """
    gt = gt.astype(int)
    segs = extract_segments(gt)
    out = {"num_segments": len(segs)}
    alert = (scores > thresh).astype(int)

    for k in ks:
        detected = 0
        leads = []
        for (s, e) in segs:
            left = max(0, s - k)
            right = s - 1
            if right < left:
                continue
            idx = np.where(alert[left:right + 1] == 1)[0]
            if idx.size > 0:
                t_star = left + int(idx[0])
                detected += 1
                leads.append(s - t_star)
        out[f"EDR@{k}"] = detected / len(segs) if len(segs) > 0 else 0.0
        out[f"AvgLead@{k}"] = float(np.mean(leads)) if len(leads) > 0 else None

    return out


def detection_adjustment(pred, gt):
    """
    repo's segment adjustment behavior (issue #14)
    """
    pred = pred.copy()
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                if pred[j] == 0:
                    pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                if pred[j] == 0:
                    pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return pred


def my_kl_loss(p, q):
    # 计算 KL散度  衡量series- attention和 prior-attention之间差异 关联差异
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    # 调整学习率
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    # 监控验证集上的性能, 在patience轮次内没有改善 就停止训练 并保存目前为止最好的模型
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience # 
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.val_loss2_min = np.inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        '''
        每个epoch结束后调用一次
        '''
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True # 循例主循环停止
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # 保存模型参数
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val',
                                              dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre',
                                              dataset=self.dataset)

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

    def build_model(self):
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()

    # 加的
    def _infer_step_for_loader(self, loader):
        ds = loader.dataset
        # your loaders: ds.mode in {"train","val","test","thre"}
        # IMPORTANT: in 'thre', slicing is non-overlapping windows => step = win_size
        if hasattr(ds, "mode") and ds.mode not in ("train", "val", "test"):
            return self.win_size
        # normal case
        return getattr(ds, "step", 1)
    # 加的   
    def _get_series_length(self, loader):
        ds = loader.dataset
        mode = getattr(ds, "mode", "test")
        if mode == "train":
            return ds.train.shape[0]
        if mode == "val":
            return ds.val.shape[0]
        # test or thre
        return ds.test.shape[0]
    
    # 加的
    def _get_gt_series(self):
        """
        Use the full test_labels from dataset (point-wise ground truth).
        Works for PSM/MSL/SMAP/SMD loaders you showed.
        """
        ds = self.test_loader.dataset
        gt = ds.test_labels
        gt = np.array(gt).reshape(-1).astype(int)
        return gt

    # 加的
    def apply_corruption(self, x, cfg):
        """
        x: torch.Tensor [B, T, C]
        cfg: dict or None
        Supported:
          - {"type":"none"}
          - {"type":"gaussian", "alpha":0.1, "channel_frac":0.3, "seed":0}
          - {"type":"missing", "rate":0.05, "impute":"zero|mean|ffill", "seed":0}
          - {"type":"spike", "rate":0.01, "factor":10.0, "channel_frac":0.1, "seed":0}
        """
        if cfg is None:
            return x
        ctype = cfg.get("type", "none")
        if ctype == "none":
            return x

        seed = int(cfg.get("seed", 0))
        rng = np.random.default_rng(seed)

        B, T, C = x.shape

        if ctype == "gaussian":
            alpha = float(cfg.get("alpha", 0.1))
            channel_frac = float(cfg.get("channel_frac", 0.3))
            num_ch = max(1, int(round(C * channel_frac)))
            ch_idx = rng.choice(C, size=num_ch, replace=False)

            std = torch.std(x, dim=(0, 1), unbiased=False) + 1e-6
            noise = torch.zeros_like(x)
            for c in ch_idx:
                sigma = alpha * std[c]
                noise[:, :, c] = torch.randn((B, T), device=x.device) * sigma
            return x + noise

        if ctype == "missing":
            rate = float(cfg.get("rate", 0.05))
            impute = cfg.get("impute", "zero")
            total = B * T * C
            m = int(round(total * rate))
            if m <= 0:
                return x

            flat_idx = rng.choice(total, size=m, replace=False)
            mask = torch.ones(total, device=x.device, dtype=torch.bool)
            mask[torch.tensor(flat_idx, device=x.device)] = False
            mask = mask.view(B, T, C)

            x_cor = x.clone()
            x_cor[~mask] = float("nan")

            if impute == "zero":
                return torch.nan_to_num(x_cor, nan=0.0)

            if impute == "mean":
                ch_mean = torch.nanmean(x_cor, dim=(0, 1))
                nan_pos = torch.isnan(x_cor)
                x_cor[nan_pos] = ch_mean.unsqueeze(0).unsqueeze(0).expand(B, T, C)[nan_pos]
                return x_cor

            if impute == "ffill":
                # fill first step nans with 0 then forward fill
                first_nan = torch.isnan(x_cor[:, 0, :])
                x_cor[:, 0, :][first_nan] = 0.0
                for t in range(1, T):
                    nan_t = torch.isnan(x_cor[:, t, :])
                    x_cor[:, t, :][nan_t] = x_cor[:, t - 1, :][nan_t]
                return x_cor

            return torch.nan_to_num(x_cor, nan=0.0)

        if ctype == "spike":
            rate = float(cfg.get("rate", 0.01))
            factor = float(cfg.get("factor", 10.0))
            channel_frac = float(cfg.get("channel_frac", 0.1))
            num_ch = max(1, int(round(C * channel_frac)))
            ch_idx = rng.choice(C, size=num_ch, replace=False)

            total_tc = B * T
            m = int(round(total_tc * rate))
            if m <= 0:
                return x

            x_cor = x.clone()
            for c in ch_idx:
                pos = rng.choice(total_tc, size=m, replace=False)
                b_idx = pos // T
                t_idx = pos % T
                x_cor[b_idx, t_idx, c] = x_cor[b_idx, t_idx, c] * factor
            return x_cor

        return x



    def vali(self, vali_loader):
        self.model.eval()

        loss_1 = []
        loss_2 = []
        with torch.no_grad():
            for i, (input_data, _) in enumerate(vali_loader):
                input = input_data.float().to(self.device)
                output, series, prior, _ = self.model(input)
                series_loss = 0.0
                prior_loss = 0.0
                # 计算每一层的 association discrepancy
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach())) + torch.mean(
                        my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)).detach(),
                            series[u])))
                    prior_loss += (torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                        self.win_size)),
                                series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(),
                                (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                        self.win_size)))))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, input)
                loss_1.append((rec_loss - self.k * series_loss).item())
                loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):

        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)

                output, series, prior, _ = self.model(input)

                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                           self.win_size)).detach(),
                                   series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, input)

                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            # vali_loss1, vali_loss2 = self.vali(self.test_loader)
            vali_loss1, vali_loss2 = self.vali(self.vali_loader)


            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)



    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth'),
                map_location=self.device
            )
        )
        self.model.eval()
        temperature = 50

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduction='none')

        def compute_window_scores(loader, corruption_cfg=None):
            """
            Return list of window score arrays, each batch gives [B, win_size]
            """
            window_scores_batches = []
            with torch.no_grad():
                for i, (input_data, labels) in enumerate(loader):
                    x = input_data.float().to(self.device)
                    x = self.apply_corruption(x, corruption_cfg)
                    
                    output, series, prior, _ = self.model(x)

                    # per-time-step recon loss: [B, T]
                    loss = torch.mean(criterion(x, output), dim=-1)

                    # discrepancy -> metric weights
                    series_loss = 0.0
                    prior_loss = 0.0
                    for u in range(len(prior)):
                        prior_norm = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(
                        1, 1, 1, self.win_size
                        )
                        if u == 0:
                            series_loss = my_kl_loss(series[u], prior_norm.detach()) * temperature
                            prior_loss = my_kl_loss(prior_norm, series[u].detach()) * temperature
                        else:
                            series_loss += my_kl_loss(series[u], prior_norm.detach()) * temperature
                            prior_loss += my_kl_loss(prior_norm, series[u].detach()) * temperature
                    metric = torch.softmax((-series_loss - prior_loss), dim=-1)  # [B, T]
                    score = (metric * loss).detach().cpu().numpy()               # [B, T]
                    window_scores_batches.append(score)
            return window_scores_batches

        def aggregate_to_series(loader, window_scores_batches):
            """
            Aggregate window scores to point-wise series score by mean over overlaps.
            Uses sequential order of windows in the loader (shuffle must be False).
            """
            step = self._infer_step_for_loader(loader)
            T_total = self._get_series_length(loader)

            sum_score = np.zeros(T_total, dtype=np.float64)
            cnt = np.zeros(T_total, dtype=np.int64)

            # count windows (sample-level) in streaming order
            n_seen = 0
            for batch_scores in window_scores_batches:
                B = batch_scores.shape[0]
                for b in range(B):
                    start = n_seen * step
                    end = start + self.win_size
                    if end > T_total:
                        break
                    sum_score[start:end] += batch_scores[b]
                    cnt[start:end] += 1
                    n_seen += 1

            cnt = np.maximum(cnt, 1)
            return (sum_score / cnt).astype(np.float32)

        # ---------- threshold: build a non-shuffled train loader for clean protocol ----------
        # original train_loader is shuffle=True, recreate it with shuffle=False
        from torch.utils.data import DataLoader
        train_ds = self.train_loader.dataset
        train_loader_noshuffle = DataLoader(
            dataset=train_ds, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        train_win_scores = compute_window_scores(train_loader_noshuffle, corruption_cfg={"type": "none"})
        train_score_series = aggregate_to_series(train_loader_noshuffle, train_win_scores)

        thresh = np.percentile(train_score_series, 100 - self.anormly_ratio)
        print("Threshold (train-only, point-wise):", thresh)

        # ---------- clean test ----------
        gt = self._get_gt_series()  # point-wise GT on test set length
        test_win_scores = compute_window_scores(self.test_loader, corruption_cfg={"type": "none"})
        test_score_series = aggregate_to_series(self.test_loader, test_win_scores)

        pred = (test_score_series > thresh).astype(int)
        gt = gt[:len(test_score_series)].astype(int)  # safety

        pred_adj = detection_adjustment(pred, gt)

        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        acc = accuracy_score(gt, pred_adj)
        precision, recall, f1, _ = precision_recall_fscore_support(gt, pred_adj, average='binary')

        auc = None
        try:
            auc = float(roc_auc_score(gt, test_score_series))
        except Exception:
            auc = None

        early = early_detection_metrics(test_score_series, gt, thresh, ks=(5, 10, 20))

        print("\n--- CLEAN TEST (point-wise) ---")
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F1 : {:0.4f}".format(
            acc, precision, recall, f1))
        print("ROC-AUC  :", auc)
        print("Early segments:", early["num_segments"])
        for k in (5, 10, 20):
            print(f"EDR@{k}: {early[f'EDR@{k}']:.4f}, AvgLead@{k}: {early[f'AvgLead@{k}']}")

        clean_res = {"acc": float(acc), "precision": float(precision), "recall": float(recall), "f1": float(f1), "auc": auc, "early": early}

        # ---------- robustness suite (fixed threshold) ----------
        corruption_suite = [
            {"type": "gaussian", "alpha": 0.05, "channel_frac": 0.3, "seed": 0},
            {"type": "gaussian", "alpha": 0.10, "channel_frac": 0.3, "seed": 0},
            {"type": "gaussian", "alpha": 0.20, "channel_frac": 0.3, "seed": 0},

            {"type": "missing", "rate": 0.01, "impute": "zero", "seed": 0},
            {"type": "missing", "rate": 0.05, "impute": "zero", "seed": 0},
            {"type": "missing", "rate": 0.10, "impute": "zero", "seed": 0},

            {"type": "missing", "rate": 0.01, "impute": "mean", "seed": 0},
            {"type": "missing", "rate": 0.05, "impute": "mean", "seed": 0},
            {"type": "missing", "rate": 0.10, "impute": "mean", "seed": 0},

            {"type": "missing", "rate": 0.01, "impute": "ffill", "seed": 0},
            {"type": "missing", "rate": 0.05, "impute": "ffill", "seed": 0},
            {"type": "missing", "rate": 0.10, "impute": "ffill", "seed": 0},

            {"type": "spike", "rate": 0.005, "factor": 5.0,  "channel_frac": 0.1, "seed": 0},
            {"type": "spike", "rate": 0.005, "factor": 10.0, "channel_frac": 0.1, "seed": 0},
            {"type": "spike", "rate": 0.010, "factor": 10.0, "channel_frac": 0.1, "seed": 0},
        ]

        for cfg in corruption_suite:
            win_scores = compute_window_scores(self.test_loader, corruption_cfg=cfg)
            score_series = aggregate_to_series(self.test_loader, win_scores)

            pred = (score_series > thresh).astype(int)
            pred_adj = detection_adjustment(pred, gt)

            acc_c = accuracy_score(gt, pred_adj)
            p_c, r_c, f1_c, _ = precision_recall_fscore_support(gt, pred_adj, average='binary')

            auc_c = None
            try:
                auc_c = float(roc_auc_score(gt, score_series))
            except Exception:
                auc_c = None

            early_c = early_detection_metrics(score_series, gt, thresh, ks=(5, 10, 20))

            print(f"\n--- CORRUPT TEST | {cfg} ---")
            print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F1 : {:0.4f}".format(
                acc_c, p_c, r_c, f1_c))
            print("ROC-AUC  :", auc_c)
            for k in (5, 10, 20):
                print(f"EDR@{k}: {early_c[f'EDR@{k}']:.4f}, AvgLead@{k}: {early_c[f'AvgLead@{k}']}")

            print("ΔF1 (corrupt - clean): {:0.4f}".format(float(f1_c) - clean_res["f1"]))

        return clean_res
