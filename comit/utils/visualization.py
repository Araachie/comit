import mediapy
import numpy as np
import torch
from torchvision.utils import draw_bounding_boxes


def to_image(x):
    return x.clamp(-1, 1).add(1.0).mul(127.5).permute(1, 2, 0).cpu().numpy().astype(np.uint8)


def show_images_with_crops(sample, reconstructions, crops):
    for i in range(len(reconstructions) - 1):
        if crops[i] == 0:
            reconstructions[i] = (
                draw_bounding_boxes(
                    (reconstructions[i].clamp(-1, 1).add(1.0).mul(127.5).to(torch.uint8)),
                    boxes=torch.tensor([[0, 0, 256, 256]]),
                    colors="red",
                    width=4,
                )
                .float()
                .div(127.5)
                .sub(1.0)
            )
        else:
            reconstructions[i] = (
                draw_bounding_boxes(
                    (reconstructions[i].clamp(-1, 1).add(1.0).mul(127.5).to(torch.uint8)),
                    boxes=torch.tensor(
                        [
                            [
                                80 * ((crops[i] - 1) % 3),
                                80 * ((crops[i] - 1) // 3),
                                80 * ((crops[i] - 1) % 3) + 96,
                                80 * ((crops[i] - 1) // 3) + 96,
                            ]
                        ]
                    ),
                    colors="red",
                    width=4,
                )
                .float()
                .div(127.5)
                .sub(1.0)
            )

    mediapy.show_images([to_image(sample.float())] + [to_image(r) for r in reconstructions], columns=6)
