import torch

from gpt.model import GPT
from gpt.config import Config



def load_model(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = Config(**ckpt["cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval(), cfg



def select_next_token(logits, temperature=1.0, k=None, generator=None):
    # logits: (V,)  - one row, over whole vocabulary of tokens
    assert temperature>0, "temperature must be positive"
    if k is not None:
        assert isinstance(k, int), "k must be an integer"
        k = min(k, logits.size(-1))
    else:
        k = logits.size(-1)
    top_values, top_indices = torch.topk(logits, k=k, dim=-1)
    probas = torch.softmax(top_values/temperature, dim=-1)
    sampled_index = torch.multinomial(probas, num_samples=1, generator=generator)
    selected_index = top_indices[sampled_index]

    return selected_index


@torch.no_grad()
def generate(model, context_len, idx, max_new_tokens, temperature, k, generator, eot_id):
    """
    Generates token sequence from a prompt through iterated sampling
    Args:
    - model: pretrained model; context_len: max window of tokens the model can see
    - idx: prompt as token ids, shape (1, T0), dtype long, on the model's device
    - max_new_tokens: max amount of tokens to generate, ending with eot_id
    - temperature, k, generator: sampling params
    Returns:
    - output: full sequence, including prompt, shape (1, T0 + n_new)
    """
    output = idx.clone()
    for _ in range(max_new_tokens):
        context = output[:, -context_len:] # shape (1, context_len)
        logits, _ = model(context, targets=None) # (1, T_window, V)
        next_tok = select_next_token(logits[0,-1], temperature, k, generator).view(1, 1)
        output = torch.cat([output, next_tok], dim=-1)
        if next_tok.item() == eot_id:
            break
    return output 