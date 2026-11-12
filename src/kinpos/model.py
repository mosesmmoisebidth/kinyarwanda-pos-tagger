"""Tagger = pretrained encoder (+ char CNN + affix/shape embeddings) -> MLP -> softmax or CRF."""
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from .crf import LinearCRF
from .features import AFFIX_BUCKETS, N_AFFIX, N_SHAPES


class CharCNN(nn.Module):
    def __init__(self, n_chars, emb_dim=48, widths=(2, 3, 4, 5), n_filters=64):
        super().__init__()
        self.emb = nn.Embedding(n_chars, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, n_filters, k, padding=k // 2) for k in widths])
        self.out_dim = n_filters * len(widths)

    def forward(self, chars):                       # chars (B, W, L)
        B, W, L = chars.shape
        flat = chars.reshape(B * W, L)
        x = self.emb(flat).transpose(1, 2)          # (BW, E, L)
        valid = (flat != 0).unsqueeze(1)            # (BW, 1, L)
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(x))[:, :, :L]
            h = h.masked_fill(~valid, float("-inf"))
            pooled.append(h.max(dim=2).values)
        out = torch.cat(pooled, dim=1)
        out = torch.nan_to_num(out, neginf=0.0)     # padded words (no chars) -> zeros
        return out.reshape(B, W, -1)


class ScalarMix(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_layers))
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, layers):
        w = torch.softmax(self.weights, dim=0)
        return self.gamma * sum(wi * li for wi, li in zip(w, layers))


class Tagger(nn.Module):
    def __init__(self, model_name, num_tags, n_chars, use_char=False, use_affix=False,
                 use_crf=False, pool="first", mix_layers=0, head_hidden=0, dropout=0.2,
                 local_files_only=False, pretrained=True, encoder_dtype=None):
        super().__init__()
        cfg = AutoConfig.from_pretrained(model_name, local_files_only=local_files_only)
        if pretrained:
            self.encoder = AutoModel.from_pretrained(model_name, config=cfg, add_pooling_layer=False,
                                                     local_files_only=local_files_only)
        else:  # weights come from a saved tagger checkpoint; only the architecture is needed
            # encoder_dtype (e.g. bfloat16) allocates the encoder directly at that precision,
            # so CPU serving never holds a full fp32 copy in RAM.
            kw = {"dtype": encoder_dtype} if encoder_dtype is not None else {}
            self.encoder = AutoModel.from_config(cfg, add_pooling_layer=False, **kw)
        self.pool = pool
        self.mix = ScalarMix(mix_layers) if mix_layers > 0 else None
        self.mix_layers = mix_layers
        d = cfg.hidden_size
        self.char_cnn = CharCNN(n_chars) if use_char else None
        if use_char:
            d += self.char_cnn.out_dim
        self.affix_emb = nn.Embedding(AFFIX_BUCKETS, 32, padding_idx=0) if use_affix else None
        self.shape_emb = nn.Embedding(N_SHAPES, 16, padding_idx=0) if use_affix else None
        if use_affix:
            d += 32 * N_AFFIX + 16
        self.dropout = nn.Dropout(dropout)
        if head_hidden > 0:
            self.head = nn.Sequential(nn.Linear(d, head_hidden), nn.GELU(), nn.Dropout(dropout),
                                      nn.Linear(head_hidden, num_tags))
        else:
            self.head = nn.Linear(d, num_tags)
        self.crf = LinearCRF(num_tags) if use_crf else None

    def encoder_parameters(self):
        return list(self.encoder.parameters())

    def head_parameters(self):
        enc = {id(p) for p in self.encoder.parameters()}
        return [p for p in self.parameters() if id(p) not in enc]

    def emissions(self, batch):
        out = self.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                           output_hidden_states=self.mix is not None)
        if self.mix is not None:
            hidden = self.mix(out.hidden_states[-self.mix_layers:])
        else:
            hidden = out.last_hidden_state
        hidden = hidden.float()                                        # (B, T, H)
        if self.pool == "mean":
            W = batch["word_mask"].shape[1]
            sub2word = batch["sub2word"]                               # (B, T), -1 = none
            onehot = (sub2word.unsqueeze(1) == torch.arange(W, device=hidden.device).view(1, W, 1)).float()
            denom = onehot.sum(-1, keepdim=True).clamp(min=1.0)
            words = torch.bmm(onehot, hidden) / denom                  # (B, W, H)
        else:
            idx = batch["first_idx"].unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
            words = hidden.gather(1, idx)
        feats = [words]
        if self.char_cnn is not None:
            feats.append(self.char_cnn(batch["chars"]))
        if self.affix_emb is not None:
            a = self.affix_emb(batch["affix"])                         # (B, W, K, 32)
            feats.append(a.flatten(2))
            feats.append(self.shape_emb(batch["shape"]))
        x = self.dropout(torch.cat(feats, dim=-1))
        return self.head(x)                                            # (B, W, num_tags)

    def loss(self, batch):
        emis = self.emissions(batch).float()    # CRF / softmax in fp32 even under fp16 autocast
        mask = batch["word_mask"]
        tags = batch["tags"]
        weight = batch["weight"]                                       # (B,)
        if self.crf is not None:
            nll = self.crf.neg_log_likelihood(emis, tags, mask)        # (B,)
            return (nll * weight).sum() / (mask.sum(1).float() * weight).sum().clamp(min=1.0)
        ce = nn.functional.cross_entropy(emis.transpose(1, 2), tags.masked_fill(~mask, -100),
                                         ignore_index=-100, reduction="none")  # (B, W)
        w = weight.unsqueeze(1) * mask.float()
        return (ce * w).sum() / w.sum().clamp(min=1.0)

    @torch.no_grad()
    def predict(self, batch):
        emis = self.emissions(batch).float()
        mask = batch["word_mask"]
        if self.crf is not None:
            return self.crf.decode(emis, mask), emis
        lengths = mask.sum(1).tolist()
        best = emis.argmax(-1)
        return [best[b, :lengths[b]].tolist() for b in range(best.shape[0])], emis
