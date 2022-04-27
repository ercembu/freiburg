from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def attention_function(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dk: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate attention function given in Vasvani et al. 2017.

    Args:
        q: The query tensor with shape (batch_size, vocab_size, dk)
        k: The key tensor with shape (batch_size, sequence_length, dk)
        v: The value tensor with shape (batch_size, sequence_length, dk)
        dk: The embedding size

    Returns:
        Tuple of:
            A torch.Tensor of the attention with shape (batch_size, vocab_size, output)
            A torch.Tensor of the attention weights with shape (batch_size, vocab_size, dk)
    """
    # Calculate attention and attention weights
    # Use torch.bmm for batch matrix multiplication.
    # START TODO #############
    attention_weights = F.softmax(torch.bmm(q, k.transpose(1, 2)) / np.sqrt(dk), dim=1)
    attention = torch.bmm(attention_weights, v)
    # END TODO #############
    return attention, attention_weights
