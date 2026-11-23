"""CNN-RNN image captioning model in the style of Show and Tell (Vinyals et al., 2015).

A frozen ImageNet ResNet-18 encodes the image into a single embedding vector,
which is fed as the first input step of an LSTM language model that predicts
the caption one token at a time.
"""
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    """Frozen ResNet-18 backbone followed by a trainable linear projection.

    Args:
        embed_size: Size of the output image embedding (must match the
            decoder's word embedding size).
    """

    def __init__(self, embed_size: int):
        super().__init__()
        # ResNet-18: a light backbone that trains quickly; only `embed` is learned.
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]  # drop the ImageNet classifier
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images.

        Args:
            images: Normalized images of shape ``(batch, 3, 224, 224)``.

        Returns:
            Image embeddings of shape ``(batch, embed_size)``.
        """
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    """LSTM language model conditioned on an image embedding.

    Args:
        embed_size: Size of the word (and image) embeddings.
        hidden_size: Number of features in the LSTM hidden state.
        vocab_size: Number of tokens in the vocabulary.
        num_layers: Number of stacked LSTM layers.
    """

    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int, num_layers: int = 1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """Teacher-forced forward pass used during training.

        The image embedding is the first time step; the caption tokens (minus
        the final ``<end>``) follow, so step ``t`` predicts token ``t``.

        Args:
            features: Image embeddings of shape ``(batch, embed_size)``.
            captions: Token ids of shape ``(batch, seq_len)``, starting with
                ``<start>`` and ending with ``<end>``.

        Returns:
            Vocabulary scores of shape ``(batch, seq_len, vocab_size)``.
        """
        embeddings = self.embed(captions[:, :-1])
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

    def sample(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        max_len: int = 20,
    ) -> List[int]:
        """Greedy decoding for a single image.

        Args:
            inputs: Image embedding of shape ``(1, 1, embed_size)``.
            states: Optional initial LSTM ``(h, c)`` state.
            max_len: Number of tokens to generate.

        Returns:
            A list of ``max_len`` token ids (may include tokens after ``<end>``;
            truncate when converting to text).
        """
        predicted_sentence = []
        for _ in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            predicted_sentence.append(predicted.item())
            inputs = self.embed(predicted).unsqueeze(1)
        return predicted_sentence


def load_trained(
    encoder_path: Path,
    decoder_path: Path,
    vocab_size: int,
    embed_size: int,
    hidden_size: int,
    device: torch.device,
) -> Tuple[EncoderCNN, DecoderRNN]:
    """Build the encoder/decoder, load saved weights and put them in eval mode.

    Weights are loaded with ``map_location`` so checkpoints saved on a CUDA
    machine also load on MPS or CPU.
    """
    encoder = EncoderCNN(embed_size)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device, weights_only=True))
    return encoder.to(device).eval(), decoder.to(device).eval()
