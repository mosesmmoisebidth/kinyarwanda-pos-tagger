"""Linear-chain CRF (batch first). Masks must be left-aligned: mask[:, 0] is True."""
import torch
import torch.nn as nn


class LinearCRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.start = nn.Parameter(torch.zeros(num_tags))
        self.end = nn.Parameter(torch.zeros(num_tags))
        self.trans = nn.Parameter(torch.zeros(num_tags, num_tags))  # trans[i, j]: i -> j

    def neg_log_likelihood(self, emis, tags, mask):
        """Per-sequence NLL. emis (B, W, T) float32, tags (B, W) long, mask (B, W) bool."""
        B, W, _ = emis.shape
        maskf = mask.float()
        tags = tags.masked_fill(~mask, 0)
        score = self.start[tags[:, 0]] + emis[:, 0].gather(1, tags[:, :1]).squeeze(1)
        for t in range(1, W):
            step = self.trans[tags[:, t - 1], tags[:, t]] + emis[:, t].gather(1, tags[:, t:t + 1]).squeeze(1)
            score = score + step * maskf[:, t]
        last = tags.gather(1, (mask.sum(1) - 1).clamp(min=0).unsqueeze(1)).squeeze(1)
        score = score + self.end[last]

        alpha = self.start.unsqueeze(0) + emis[:, 0]
        for t in range(1, W):
            nxt = torch.logsumexp(alpha.unsqueeze(2) + self.trans.unsqueeze(0) + emis[:, t].unsqueeze(1), dim=1)
            alpha = torch.where(mask[:, t].unsqueeze(1), nxt, alpha)
        log_z = torch.logsumexp(alpha + self.end.unsqueeze(0), dim=1)
        return log_z - score

    @torch.no_grad()
    def decode(self, emis, mask):
        B, W, T = emis.shape
        score = self.start.unsqueeze(0) + emis[:, 0]
        history = []
        for t in range(1, W):
            cand = score.unsqueeze(2) + self.trans.unsqueeze(0)          # (B, T_prev, T_next)
            best, idx = cand.max(dim=1)
            nxt = best + emis[:, t]
            m = mask[:, t].unsqueeze(1)
            score = torch.where(m, nxt, score)
            # where masked, point each tag back to itself so the path is unchanged
            idx = torch.where(m, idx, torch.arange(T, device=emis.device).unsqueeze(0).expand(B, T))
            history.append(idx)
        score = score + self.end.unsqueeze(0)
        best_last = score.argmax(dim=1)
        paths = [best_last]
        for idx in reversed(history):
            best_last = idx.gather(1, best_last.unsqueeze(1)).squeeze(1)
            paths.append(best_last)
        paths.reverse()
        out = torch.stack(paths, dim=1)                                   # (B, W)
        lengths = mask.sum(1).tolist()
        return [out[b, :lengths[b]].tolist() for b in range(B)]
