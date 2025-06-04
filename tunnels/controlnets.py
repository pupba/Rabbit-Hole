from cores.types import IO
from typing import Tuple


def apply_controlnet(
    positive: IO.CONDITIONING,
    negative: IO.CONDITIONING,
    control_net: IO.CONTROL_NET,
    image: IO.IMAGE,
    strength: float = 1.000,
    start_percent: float = 0.000,
    end_percent: float = 1.000,
    vae: IO.VAE = None,
    extra_concat=[],
) -> Tuple[IO.CONDITIONING, IO.CONDITIONING]:
    """
    Applies a ControlNet model to both positive and negative conditioning, returning updated conditionings for use in diffusion models.

    Args:
        positive (IO.CONDITIONING): The positive conditioning to which ControlNet will be applied.
        negative (IO.CONDITIONING): The negative conditioning to which ControlNet will be applied.
        control_net (IO.CONTROL_NET): The ControlNet model instance.
        image (IO.IMAGE): The input image used as the ControlNet hint (e.g., edge map, pose, depth).
        strength (float, optional): The strength of the ControlNet guidance (default: 1.0).
        start_percent (float, optional): The start percent for applying ControlNet (default: 0.0).
        end_percent (float, optional): The end percent for applying ControlNet (default: 1.0).
        vae (IO.VAE, optional): An optional VAE object for advanced conditioning.
        extra_concat (list, optional): Extra concatenation tensors for multi-modal conditioning (default: empty list).

    Returns:
        Tuple[IO.CONDITIONING, IO.CONDITIONING]: The updated positive and negative conditionings, with ControlNet guidance applied.

    Notes:
        - If `strength` is set to 0, the function returns the original (positive, negative) conditioning without modification.
        - This function modifies the conditioning data structures to embed ControlNet hints and control parameters.
        - Both positive and negative conditionings are processed, enabling use in classifier-free guidance pipelines.

    Example:
        >>> pos_c, neg_c = apply_controlnet(
        ...     positive, negative, controlnet, hint_img,
        ...     strength=0.8, start_percent=0.2, end_percent=0.8, vae=vae
        ... )
    """
    if strength == 0:
        return (positive, negative)

    control_hint = image.movedim(-1, 1)
    cnets = {}

    out = []
    for conditioning in [positive, negative]:
        c = []
        for t in conditioning:
            d = t[1].copy()

            prev_cnet = d.get("control", None)
            if prev_cnet in cnets:
                c_net = cnets[prev_cnet]
            else:
                c_net = control_net.copy().set_cond_hint(
                    control_hint,
                    strength,
                    (start_percent, end_percent),
                    vae=vae,
                    extra_concat=extra_concat,
                )
                c_net.set_previous_controlnet(prev_cnet)
                cnets[prev_cnet] = c_net

            d["control"] = c_net
            d["control_apply_to_uncond"] = False
            n = [t[0], d]
            c.append(n)
        out.append(c)
    return (out[0], out[1])
