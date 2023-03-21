# A Transformer-based classification for planetary transit signals

This code contains the implementation of the model used in my paper.


The architecture contains three encoders. Each encoder works with the `EncoderBlock`  class.  This class calls the `MultiHeadedAttention` class implemented in `model.py`, but it is also possible to work with pytorch's own class (`nn.MultiheadAttention`). 

If you decide to work with pytorchs's class (`nn.MultiheadAttention`), the important is to modify the return so that it returns the attention weights. Also, change the return to obtain the weights of each head:

For example:

before:
```python 
return attn_output, attn_output_weights.sum(dim=1) / num_heads
```


after:
return attn_output, attn_output_weights


Diagram of the designed architecture:

<img src='imgs/model_exo.png' width='800'>

## Requirements

- Python 3.6.8
- Torch 1.8
- numpy 1.19
- commet_ml 3.8.1
