"""Turn sentences into padded tensors. Long sentences are split into word chunks that fit."""
import torch

from .data import TAG2ID
from .features import affix_ids, char_ids, shape_id


class Encoded:
    __slots__ = ("sent_id", "offset", "words", "input_ids", "sub2word", "first_idx",
                 "chars", "affix", "shape", "tags", "weight")


def encode_sentences(sents, tokenizer, char_vocab, max_len=512, weight=1.0, id_offset=0):
    """Returns a list of Encoded chunks; chunk.offset is the index of its first word."""
    out = []
    budget = max_len - 2
    for si, s in enumerate(sents):
        words = s["words"]
        pieces = tokenizer(list(words), add_special_tokens=False)["input_ids"]
        start = 0
        while start < len(words):
            end, used = start, 0
            while end < len(words) and (used + max(len(pieces[end]), 1) <= budget or end == start):
                used += max(len(pieces[end]), 1)
                end += 1
            e = Encoded()
            e.sent_id = si + id_offset
            e.offset = start
            e.words = words[start:end]
            ids = [tokenizer.cls_token_id]
            sub2word = [-1]
            first = []
            for wi, p in enumerate(pieces[start:end]):
                p = p[:budget] if p else [tokenizer.unk_token_id]
                first.append(len(ids))
                ids.extend(p)
                sub2word.extend([wi] * len(p))
            ids = ids[:max_len - 1] + [tokenizer.sep_token_id]
            sub2word = sub2word[:max_len - 1] + [-1]
            first = [min(f, max_len - 2) for f in first]
            e.input_ids = ids
            e.sub2word = sub2word
            e.first_idx = first
            e.chars = [char_ids(w, char_vocab) for w in e.words]
            e.affix = [affix_ids(w) for w in e.words]
            e.shape = [shape_id(w) for w in e.words]
            e.tags = [TAG2ID[t] for t in s["tags"][start:end]] if s["tags"] else [0] * (end - start)
            e.weight = float(s.get("weight", weight))
            out.append(e)
            start = end
    return out


def collate(chunks, pad_id):
    B = len(chunks)
    T = max(len(c.input_ids) for c in chunks)
    W = max(len(c.words) for c in chunks)
    L = max(len(x) for c in chunks for x in c.chars)
    K = len(chunks[0].affix[0])
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.long)
    sub2word = torch.full((B, T), -1, dtype=torch.long)
    first_idx = torch.zeros((B, W), dtype=torch.long)
    chars = torch.zeros((B, W, L), dtype=torch.long)
    affix = torch.zeros((B, W, K), dtype=torch.long)
    shape = torch.zeros((B, W), dtype=torch.long)
    tags = torch.zeros((B, W), dtype=torch.long)
    word_mask = torch.zeros((B, W), dtype=torch.bool)
    weight = torch.ones(B)
    for b, c in enumerate(chunks):
        n, w = len(c.input_ids), len(c.words)
        input_ids[b, :n] = torch.tensor(c.input_ids)
        attn[b, :n] = 1
        sub2word[b, :n] = torch.tensor(c.sub2word)
        first_idx[b, :w] = torch.tensor(c.first_idx)
        for wi, ch in enumerate(c.chars):
            chars[b, wi, :len(ch)] = torch.tensor(ch)
        affix[b, :w] = torch.tensor(c.affix)
        shape[b, :w] = torch.tensor(c.shape)
        tags[b, :w] = torch.tensor(c.tags)
        word_mask[b, :w] = True
        weight[b] = c.weight
    return {"input_ids": input_ids, "attention_mask": attn, "sub2word": sub2word,
            "first_idx": first_idx, "chars": chars, "affix": affix, "shape": shape,
            "tags": tags, "word_mask": word_mask, "weight": weight}


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}
