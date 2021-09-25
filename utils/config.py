from dataclasses import dataclass

@dataclass
class Config:
    num_heads = 5
    max_len = 200
    top_k = 0
    top_p = .92