import numpy as np
from numpy.testing import assert_allclose

from models.gp.common import PCAPipeline, PCAPipelineConfig


def test_project_latent_variance_matches_manual_projection():
    data = np.array(
        [
            [2.0, 0.0],
            [-2.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=np.float64,
    )
    pipeline = PCAPipeline(PCAPipelineConfig(n_components=2))
    pipeline.fit(data)

    latent_var = np.array([0.25, 0.5], dtype=np.float64)
    expected = latent_var @ np.square(pipeline.components_)
    projected = pipeline.project_variance(latent_var)
    assert_allclose(projected, expected)

    scale = np.array([1.2, 0.8], dtype=np.float64)
    projected_scaled = pipeline.project_variance(latent_var, scale=scale)
    assert_allclose(projected_scaled, expected * np.square(scale))


def test_project_latent_variance_batch_input():
    data = np.random.RandomState(0).normal(size=(16, 3))
    pipeline = PCAPipeline(PCAPipelineConfig(n_components=3))
    pipeline.fit(data)

    latent_var = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
        dtype=np.float64,
    )
    expected = latent_var @ np.square(pipeline.components_)
    projected = pipeline.project_variance(latent_var)
    assert_allclose(projected, expected)
