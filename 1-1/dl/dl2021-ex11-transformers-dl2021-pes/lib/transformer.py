import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class TransformerModel(nn.Module):
    def __init__(
        self,
        en_vocab_size,
        de_vocab_size,
        max_len=20,
        hidden=64,
        enc_layers=6,
        dec_layers=6,
        heads=4,
    ):
        super().__init__()
        """Initializes the TransformerModel.

        Args:
            en_vocab_size    : Size of the English vocabulary used in this exercise.
            de_vocab_size    : Size of the German vocabulary used in this exercise.
            max_len          : Maximum number of words in the input sentence.
                               Notice that during training the input to the decoder
                               will be shifted one token to the right starting with <sos>,
                               and will therefore have a maximum length of max_len + 1.
            hidden           : Size of the hidden space.
            enc_layers       : Number of encoder blocks.
            dec_layers       : Number of decoder blocks.
            heads            : Number of heads in Multi-head attention blocks.
        """

        self.max_len = max_len
        self.en_vocab_size = en_vocab_size
        self.de_vocab_size = de_vocab_size
        self.hidden = hidden
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.heads = heads

        # Linear transformation used for embedding the input to the encoder
        self.enc_input_dense = nn.Linear(self.de_vocab_size, self.hidden)

        # (Learned) positional encoding to be added to the embedded encoder input
        self.enc_pos_enc = nn.Parameter(torch.zeros((1, self.max_len, self.hidden)))

        # List of hidden layers in the feed-forward network of different encoder blocks
        self.enc_increase_hidden = nn.ModuleList(
            [nn.Linear(self.hidden, self.hidden * 2) for i in range(self.enc_layers)]
        )

        # List of output layers used in the feed-forward network of different encoder blocks
        self.enc_decrease_hidden = nn.ModuleList(
            [nn.Linear(self.hidden * 2, self.hidden) for i in range(self.enc_layers)]
        )

        # List of final layer normalizations used in different encoder blocks
        self.enc_layer_norm = nn.ModuleList(
            [nn.LayerNorm(self.hidden) for i in range(self.enc_layers)]
        )

        # List of Multi-head attention blocks used in different encoder blocks
        self.enc_att = nn.ModuleList(
            [
                MultiHeadAttention(self.heads, self.hidden)
                for i in range(self.enc_layers)
            ]
        )

        # Linear transformation used for embedding the input to the decoder
        self.dec_input_dense = nn.Linear(self.en_vocab_size, self.hidden)

        # (Learned) positional encoding to be added to the embedded decoder input
        self.dec_pos_enc = nn.Parameter(torch.zeros((1, self.max_len + 1, self.hidden)))

        # List of hidden layers in the feed-forward network of different decoder blocks
        self.dec_increase_hidden = nn.ModuleList(
            [nn.Linear(self.hidden, self.hidden * 2) for i in range(self.dec_layers)]
        )

        # List of output layers in the feed-forward network of different decoder blocks
        self.dec_decrease_hidden = nn.ModuleList(
            [nn.Linear(self.hidden * 2, self.hidden) for i in range(self.dec_layers)]
        )

        # List of final layer normalizations used in different decoder blocks
        self.dec_layer_norm = nn.ModuleList(
            [nn.LayerNorm(self.hidden) for i in range(self.dec_layers)]
        )

        # List of (first) Multi-head attention blocks used in different decoder blocks
        self.dec_att = nn.ModuleList(
            [
                MultiHeadAttention(self.heads, self.hidden)
                for i in range(self.dec_layers)
            ]
        )

        # List of (second) Multi-head attention blocks used in different decoder blocks
        self.enc_dec_att = nn.ModuleList(
            [
                MultiHeadAttention(self.heads, self.hidden)
                for i in range(self.dec_layers)
            ]
        )

        # Final fully connected layer converting the final decoder output to predictions over the output vocabulary
        self.decoder_final_dense = nn.Linear(self.hidden, self.en_vocab_size)

    def forward(self, x1, x2):
        """Implement the forward pass of the Transformer model.

        Args:
            x1: with shape (batch_size, self.max_len, self.de_vocab_size)
            x2: with shape (batch_size, self.max_len + 1, self.en_vocab_size)

        Returns:
            Tuple of:
                decoding: final output of the model,
                          with shape (batch_size, self.max_len + 1, self.en_vocab_size)
                attention: attention_weights of the last multi-head attention block in decoder,
                           with shape (batch_size, self.max_len + 1, self.max_len)
        """

        # Embed inputs to hidden dimension
        # START TODO #############
        # enc_input_emb = ...
        # dec_input_emb = ...
        enc_input_emb = self.enc_input_dense(x1)
        dec_input_emb = self.dec_input_dense(x2)
        # END TODO #############

        # Add positional encodings
        # START TODO #############
        # encoding = ...
        # decoding = ...
        encoding = enc_input_emb + self.enc_pos_enc
        decoding = dec_input_emb + self.dec_pos_enc
        # END TODO #############

        # Loop over the encoder blocks
        for i in range(self.enc_layers):
            # Encoder Self-Attention
            # In the ith encoder block:
            # 1) Pass encoding through the MultiHeadAttention block
            # 2) Pass the output through the FeedForward block
            # FeedForward block consists of:
            # 1 - a Linear layer doubling the hidden size
            # 2 - a ReLU activation function
            # 3 - a Linear layer halving the hidden size
            # 3) Sum up the outputs of the previous two steps
            # 4) pass the result through the encoder layer-normalization
            # START TODO #############

            enc_att_output, enc_att_weights = self.enc_att[i](encoding, encoding, encoding)

            # Feed Forward Module
            enc_linear_1 = self.enc_increase_hidden[i](enc_att_output)
            enc_relu = F.relu(enc_linear_1)
            enc_linear_2 = self.enc_decrease_hidden[i](enc_relu)
            encoding = self.enc_layer_norm[i](enc_linear_2 + enc_att_output)

            # END TODO #############

        # Without decoder blocks, attention would be None
        attention = None

        # Loop over the decoder blocks
        for i in range(self.dec_layers):
            # Decoder Self-Attention
            # Repeat the steps in the encoder, but with three main differences:
            # 1) In each decoder block, there are two successive MultiHeadAttention blocks
            # 2) The first MultiHeadAttention is masked (you therefore need to set mask=True)
            # 3) The three inputs to the second MultiHeadAttention block are mixed:
            # two coming from the encoder, and
            # one coming from the previous MultiHeadAttention block in the decoder
            # START TODO #############

            dec_att_output, dec_att_weights = self.dec_att[i](decoding, decoding, decoding, mask=True)
            enc_dec_output, attention = self.enc_dec_att[i](dec_att_output, encoding, encoding)

            # Feed Forward Module
            dec_linear_1 = self.dec_increase_hidden[i](enc_dec_output)
            dec_relu = F.relu(dec_linear_1)
            dec_linear_2 = self.dec_decrease_hidden[i](dec_relu)
            decoding = self.dec_layer_norm[i](dec_linear_2+enc_dec_output)
            # END TODO #############

        # Map the hidden dimension of the decoder output back to self.en_vocab_size
        # START TODO #############
        decoding = self.decoder_final_dense(decoding)
        # END TODO #############

        return decoding, attention


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, h, hidden):
        super().__init__()

        self.h = h
        self.hidden = hidden

        self.W_query = nn.Parameter(
            torch.normal(mean=torch.zeros((self.hidden, self.hidden)), std=1e-2)
        )
        self.W_key = nn.Parameter(
            torch.normal(mean=torch.zeros((self.hidden, self.hidden)), std=1e-2)
        )
        self.W_value = nn.Parameter(
            torch.normal(mean=torch.zeros((self.hidden, self.hidden)), std=1e-2)
        )
        self.W_output = nn.Parameter(
            torch.normal(mean=torch.zeros((self.hidden, self.hidden)), std=1e-2)
        )

        self.layer_norm = nn.LayerNorm(self.hidden)

    def forward(self, query, key, value, mask=False):
        chunk_size = int(self.hidden / self.h)

        multi_query = torch.matmul(query, self.W_query).split(
            split_size=chunk_size, dim=-1
        )
        multi_query = torch.stack(multi_query, dim=0)

        multi_key = torch.matmul(key, self.W_key).split(split_size=chunk_size, dim=-1)
        multi_key = torch.stack(multi_key, dim=0)

        multi_value = torch.matmul(value, self.W_value).split(
            split_size=chunk_size, dim=-1
        )
        multi_value = torch.stack(multi_value, dim=0)

        scaling_factor = torch.tensor(np.sqrt(multi_query.shape[-1]))
        dotp = torch.matmul(multi_query, multi_key.transpose(2, 3)) / scaling_factor
        attention_weights = F.softmax(dotp, dim=-1)

        if mask:
            attention_weights = attention_weights.tril()
            attention_weights = attention_weights / attention_weights.sum(
                dim=3, keepdim=True
            )

        weighted_sum = torch.matmul(attention_weights, multi_value)
        weighted_sum = weighted_sum.split(1, dim=0)
        weighted_sum = torch.cat(weighted_sum, dim=-1).squeeze()

        output = weighted_sum + query
        output = self.layer_norm(output)
        return output, attention_weights
