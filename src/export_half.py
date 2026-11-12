"""Shrink a saved tagger (model.pt) to fp16 for download / CPU serving. Usage: export_half.py in.pt out.pt"""
import sys

import torch

ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
ckpt["state_dict"] = {k: (v.half() if v.is_floating_point() else v) for k, v in ckpt["state_dict"].items()}
torch.save(ckpt, sys.argv[2])
print("wrote", sys.argv[2])
