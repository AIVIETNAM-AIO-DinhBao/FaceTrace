import unittest

import torch

from src.models.residual import (
    ResidualImageModel,
    TinyResidualCNN,
    box_blur_residual,
    trainable_parameter_count,
)


class ResidualModelTests(unittest.TestCase):
    def test_constant_image_has_no_border_residual(self):
        images = torch.full((2, 3, 16, 16), 0.5, dtype=torch.float32)
        residual = box_blur_residual(images)
        self.assertTrue(torch.allclose(residual, torch.zeros_like(residual), atol=1e-6))

    def test_filter_preserves_shape_and_signed_values(self):
        images = torch.zeros((1, 3, 16, 16), dtype=torch.float32)
        images[:, :, 8:, :] = 1.0
        residual = box_blur_residual(images)
        self.assertEqual(tuple(residual.shape), tuple(images.shape))
        self.assertLess(float(residual.min()), 0.0)
        self.assertGreater(float(residual.max()), 0.0)

    def test_tiny_cnn_output_shape_and_parameter_budget(self):
        model = TinyResidualCNN()
        logits = model(torch.rand(4, 3, 224, 224))
        self.assertEqual(tuple(logits.shape), (4, 2))
        self.assertLess(trainable_parameter_count(model), 1_000_000)

    def test_rgb_and_residual_controls_have_matching_parameters(self):
        residual = ResidualImageModel(use_residual=True)
        rgb = ResidualImageModel(use_residual=False)
        self.assertEqual(trainable_parameter_count(residual), trainable_parameter_count(rgb))
        images = torch.rand(2, 3, 224, 224)
        self.assertEqual(tuple(residual(images).shape), (2, 2))
        self.assertEqual(tuple(rgb(images).shape), (2, 2))


if __name__ == "__main__":
    unittest.main()
