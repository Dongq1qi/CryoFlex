import torch

from model.loss import sinkhorn_loss


def test_sinkhorn_loss_is_differentiable_and_zero_on_identical_clouds():
    x = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        requires_grad=True,
    )
    y = x.detach().clone()

    loss = sinkhorn_loss(x, y, blur=0.2, n_iters=30)

    assert loss.abs().item() < 1e-5
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_sinkhorn_loss_defaults_match_reference_registration_settings():
    x = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    y = x + torch.tensor([0.5, -0.25])

    default_loss = sinkhorn_loss(x, y)
    explicit_reference_loss = sinkhorn_loss(
        x,
        y,
        blur=0.1,
        p=2,
        n_iters=100,
        tol=1e-3,
        debias=True,
    )

    assert torch.allclose(default_loss, explicit_reference_loss)
