# # Copyright (c) ModelScope Contributors. All rights reserved.
# from .causal_lm import CustomCrossEntropyLoss
# from .embedding import ContrastiveLoss, CosineSimilarityLoss, InfonceLoss, OnlineContrastiveLoss
# from .reranker import ListwiseRerankerLoss, PointwiseRerankerLoss

# loss_map = {
#     'cross_entropy': CustomCrossEntropyLoss,  # examples
#     # embedding
#     'cosine_similarity': CosineSimilarityLoss,
#     'contrastive': ContrastiveLoss,
#     'online_contrastive': OnlineContrastiveLoss,
#     'infonce': InfonceLoss,
#     # # reranker
#     'pointwise_reranker': PointwiseRerankerLoss,
#     'listwise_reranker': ListwiseRerankerLoss,
# }

# Copyright (c) ModelScope Contributors. All rights reserved.
from .causal_lm import CustomCrossEntropyLoss
from .embedding import ContrastiveLoss, CosineSimilarityLoss, InfonceLoss, OnlineContrastiveLoss
from .reranker import ListwiseRerankerLoss, PointwiseRerankerLoss
from .seq_cls import SmoothL1RegressionLoss, PLCCLoss, FidelityLoss, MixedLoss

loss_map = {
    'cross_entropy': CustomCrossEntropyLoss,  # examples
    # embedding
    'cosine_similarity': CosineSimilarityLoss,
    'contrastive': ContrastiveLoss,
    'online_contrastive': OnlineContrastiveLoss,
    'infonce': InfonceLoss,
    # # reranker
    'pointwise_reranker': PointwiseRerankerLoss,
    'listwise_reranker': ListwiseRerankerLoss,
    # seq_cls
    'smooth_l1_regression': SmoothL1RegressionLoss,
    'plcc': PLCCLoss,
    'fidelity': FidelityLoss,
    'mixed': MixedLoss
}
