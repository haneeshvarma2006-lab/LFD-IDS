"""Module 4 -- K-means clustering-based defence (KCD)."""

from .kcd import DefenceResult, KMeansClusteringDefence, assign_cluster_classes
from .kmeans import KMeans, KMeansResult, euclidean_distances
from .retrain import (
    CloudModelPublisher,
    DeployedModel,
    RobustnessValidator,
    RobustnessVerdict,
    select_trusted_anchors,
)
from .variants import (
    BAGKCD,
    BOOTKCD,
    DEFENCE_FOR_ATTACK,
    DEFENCE_REGISTRY,
    build_defence,
)

__all__ = [
    "BAGKCD",
    "BOOTKCD",
    "CloudModelPublisher",
    "DEFENCE_FOR_ATTACK",
    "DEFENCE_REGISTRY",
    "DefenceResult",
    "DeployedModel",
    "KMeans",
    "KMeansClusteringDefence",
    "KMeansResult",
    "RobustnessValidator",
    "RobustnessVerdict",
    "assign_cluster_classes",
    "build_defence",
    "euclidean_distances",
    "select_trusted_anchors",
]
