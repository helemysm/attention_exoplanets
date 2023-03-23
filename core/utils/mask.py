import torch

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' 
    For masking out the subsequent info. In our implementation we don't use mask. 
    In a future work we will implement different types of masks for different attention mechanisms. 
    This in order to scale to longer sequences.
    '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask