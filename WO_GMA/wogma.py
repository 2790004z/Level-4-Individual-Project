import torch
import torch.nn as nn
import torch.nn.functional as F

class WOGMA(nn.Module):
    def __init__(
        self,
        pretrained_model,  # your pretrained ST-GCN
        feature_dim=32,
        hidden_dim=128,
        num_classes=2,
        top_k_ratio=8,
        theta_class=0.4,
        theta_score=0.3,
        class_weights=None
    ):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.top_k_ratio = top_k_ratio
        self.theta_class = theta_class
        self.theta_score = theta_score

        if class_weights is not None:
            self.class_weights = torch.FloatTensor(class_weights)
        else:
            self.class_weights = None

        # CPGB conv + FC
        self.cpgb_conv = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.cpgb_fc = nn.Linear(hidden_dim, num_classes)

        # OAMB LSTM + FC
        self.oamb_lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, batch_first=True)
        self.oamb_fc = nn.Linear(hidden_dim, num_classes)

    def forward_local_extraction(self, bag):
        """
        Applies the pretrained ST-GCN to each clip in 'bag' and returns 
        a (L, feature_dim) tensor of clip features.
        """
        clip_features_list = []
        for i in range(bag.shape[0]):
            clip = bag[i:i+1]  # shape (1, C, T, V, M)
            with torch.no_grad():
                clip_logits, feats = self.pretrained_model.extract_feature(clip)
            # feats: shape (1, feature_dim, t, v)
            feats = feats.squeeze(-1)
            feats = feats.mean(dim=[2, 3])  # -> (1, feature_dim)
            clip_features_list.append(feats)
        return torch.cat(clip_features_list, dim=0)  # (L, feature_dim)

    def forward_cpgb(self, clip_features):
        """
        CPGB branch with 1D conv => clip logits => top-K aggregator => video-level logit
        """
        L = clip_features.shape[0]
        x = clip_features.unsqueeze(0).transpose(1,2)  # (1, feature_dim, L)
        x = self.cpgb_conv(x)                          # -> (1, hidden_dim, L)
        x = x.transpose(1,2)                           # -> (1, L, hidden_dim)
        clip_logits = self.cpgb_fc(x)                  # -> (1, L, num_classes)
        clip_logits = clip_logits.squeeze(0)           # -> (L, num_classes)

        # top-K aggregator => video_scores
        k = max(1, L // self.top_k_ratio)
        video_scores = []
        for c in range(self.num_classes):
            vals, idxs = torch.topk(clip_logits[:, c], k)
            video_scores.append(vals.mean())
        video_scores = torch.stack(video_scores)  # (num_classes,)
        return clip_logits, video_scores

    def forward_oamb(self, clip_features):
        """
        Online LSTM aggregator => clip logits => top-K aggregator => video-level
        """
        L = clip_features.shape[0]
        x = clip_features.unsqueeze(0)             # (1, L, feature_dim)
        y, _ = self.oamb_lstm(x)                   # -> (1, L, hidden_dim)
        clip_logits = self.oamb_fc(y)              # -> (1, L, num_classes)
        clip_logits = clip_logits.squeeze(0)       # -> (L, num_classes)

        # top-K aggregator => video_scores
        k = max(1, L // self.top_k_ratio)
        video_scores = []
        for c in range(self.num_classes):
            vals, idxs = torch.topk(clip_logits[:, c], k)
            video_scores.append(vals.mean())
        video_scores = torch.stack(video_scores)   # (num_classes,)
        return clip_logits, video_scores

    def generate_pseudo_labels(self, clip_logits, video_scores, gt_label):
        """
        Generate pseudo labels for each clip given the video-level label (gt_label).
        
        - If self.num_classes == 2 (binary):
            Matches the old logic:
            If gt_label=1 => threshold each clip's "pos" dimension (class=1).
            If gt_label=0 => remain zeros.
        
        - If self.num_classes > 2 (multi-class, single-label):
            For ground-truth class = c, threshold clip_probs[:, c] if the 
            video-level probability for class c is >= self.theta_class.
            (You could enhance this further, but this is a minimal extension.)
        """
        if self.num_classes == 2:
            # original binary logic unchanged
            pseudo = torch.zeros_like(clip_logits)
            if gt_label == 1:  
                pos_prob = torch.sigmoid(video_scores[1])
                if pos_prob >= self.theta_class:
                    clip_probs = torch.sigmoid(clip_logits[:, 1])
                    pseudo[:, 1] = (clip_probs >= self.theta_score).float()
            return pseudo
        
        else:
            # multi-class single-label approach
            pseudo = torch.zeros_like(clip_logits)  # (L, num_classes)
            # If the ground-truth label is c, treat c as “positive”
            # Check if that class's video-level prob >= theta_class
            gt_prob = torch.sigmoid(video_scores[gt_label])
            if gt_prob >= self.theta_class:
                # threshold each clip's logit for that specific class
                clip_probs = torch.sigmoid(clip_logits[:, gt_label])
                pseudo[:, gt_label] = (clip_probs >= self.theta_score).float()
            # everything else remains zero
            return pseudo

    def forward(self, bag, gt_label=None):
        device = bag.device

        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(device)

        # 1) local
        clip_features = self.forward_local_extraction(bag)

        # 2) CPGB
        clip_logits_cpgb, vid_scores_cpgb = self.forward_cpgb(clip_features)
        loss_pMIL = torch.tensor(0.0, device=device)
        
        if gt_label is not None:
            ce_fn = nn.CrossEntropyLoss(weight=self.class_weights)
            # cross-entropy with video label for the video-level score
            loss_pMIL = ce_fn(vid_scores_cpgb.unsqueeze(0), torch.tensor([gt_label], device=device))

        # 3) Pseudo label generation
        pseudo = None
        if gt_label is not None:
            pseudo = self.generate_pseudo_labels(clip_logits_cpgb, vid_scores_cpgb, gt_label)

        # 4) OAMB
        clip_logits_oamb, vid_scores_oamb = self.forward_oamb(clip_features)
        loss_FML = torch.tensor(0.0, device=device)
        loss_oMIL = torch.tensor(0.0, device=device)

        if pseudo is not None:
            # === FML ===
            if self.num_classes == 2:
                # original binary logic: BCE on flattened logits vs. pseudo
                bce_fn = nn.BCEWithLogitsLoss()
                loss_FML = bce_fn(clip_logits_oamb.view(-1), pseudo.view(-1))
            else:
                with torch.no_grad():
                    # argmax would give a valid index for row w. exactly 1 "1". 
                    # If sum=0 => that row is background => ignore
                    # If sum>1 => your thresholding might yield multiple classes => handle as ignore
                    sums = pseudo.sum(dim=1)
                    max_vals, max_ids = pseudo.max(dim=1)
                    # (L,) each element is 1 or 0 for "pos or not in that class"
                    clip_labels = torch.full_like(sums, fill_value=-1, dtype=torch.long)
                    # set clip_labels to max_ids where sums==1
                    mask = (sums == 1) & (max_vals == 1)
                    clip_labels[mask] = max_ids[mask]

                # Cross-entropy with ignore_index for those background rows
                ce_fn_multi = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=-1)
                loss_FML = ce_fn_multi(clip_logits_oamb, clip_labels)

            # === oMIL ===
            ce_fn = nn.CrossEntropyLoss(weight=self.class_weights)
            loss_oMIL = ce_fn(vid_scores_oamb.unsqueeze(0), torch.tensor([gt_label], device=device))

        loss_total = loss_pMIL + loss_FML + loss_oMIL

        return {
            "loss_total": loss_total,
            "loss_pMIL": loss_pMIL,
            "loss_FML": loss_FML,
            "loss_oMIL": loss_oMIL,
            "clip_logits_cpgb": clip_logits_cpgb,
            "clip_logits_oamb": clip_logits_oamb,
            "video_scores_cpgb": vid_scores_cpgb,
            "video_scores_oamb": vid_scores_oamb
        }