import time
from typing import Optional, Callable, Dict, Union


from logzero import logger
import torch


def gauss_newton(
    fun: Callable[[Union[Dict[str, torch.Tensor], torch.Tensor]], torch.Tensor],
    frames_to_parameters: Callable[[], Dict[str, torch.Tensor]],
    update_frames: Callable[[Dict[str, torch.Tensor]], None],
    estimate_jacobian: Callable[
        [Dict[str, torch.Tensor], Optional[torch.Tensor]], torch.Tensor
    ],
    max_ite=10,
    verbose: bool = True,
):
    """
    Gauss-Newton loop for least-squares. Parameters are given in a dictionary and
    managed inside functions

    fun: error function. It receives the dictionary of parameters and return the error.
    frames_to_numpy: function to generate a initial guess from parameters.
    update_frames: function to update the parameters after the optimization.
    max_iter: max number of iterations.
    diff_step: step for numerical jacobian.
    verbose: print the output of the optimization step by step.
    """
    keep_going = True
    it = 0
    while keep_going:
        x_o = frames_to_parameters()
        a = time.time()
        error_o = fun(x_o)
        b = time.time()

        jac = estimate_jacobian(x_o, error_o).float()

        x_flat = -(
            torch.pinverse(jac[torch.abs(error_o) > 0, :])
            @ error_o[torch.abs(error_o) > 0].float()
        )

        c = time.time()
        idx = 0
        x = dict()
        for key, x_ in x_o.items():
            x[key] = x_ + x_flat[idx : idx + x_.numel()].reshape(x_.shape)
            idx = idx + x_.numel()
        error = fun(x)
        norm = torch.dot(error, error)
        norm_o = torch.dot(error_o, error_o)
        imp = norm - norm_o
        if verbose:
            logger.info(
                f"Iteration:  {it:d}; Cost function {norm.cpu().detach().numpy():.5f}; Last Cost function {norm_o.cpu().detach().numpy():.5f} Improvement: {imp.cpu().detach().numpy():.7f} time eval : {(b - a) * 1000:.0f}ms - time jac: {(c - b) * 1000:.0f} ms"
            )
            it = it + 1
        if (abs(imp) < 1e-8) or (imp > 0) or (it >= max_ite):
            keep_going = False
        else:
            update_frames(x)
