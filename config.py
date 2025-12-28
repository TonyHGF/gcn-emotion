from dataclasses import dataclass

@dataclass
class FeatureExtractorConfig:
    """
    A minimal feature extractor configuration for adapting (B, 1, Ch, T) to (B, Ch, F).

    mode:
        - "variance": F=1, uses variance over time per channel.
        - "mean_var": F=2, uses mean and variance over time per channel.
        - "identity_time": treat time as features directly: F=T (not recommended for DGCNN, but useful for debugging).
    """
    mode: str = "variance"