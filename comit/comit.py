import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from hydra.utils import instantiate
from torch import Tensor
from torchdiffeq import odeint_adjoint as odeint

from .components.misc.apg import MomentumBuffer, adaptive_projected_guidance


class COMiTBase(nn.Module):
    def __init__(
        self,
        predictor: nn.Module,
        quantizer: nn.Module,
        vae: nn.Module | None = None,
    ):
        """
        The base class implementing the main components of COMiT

        :param predictor: Main DiT-based backbone that encodes crops into latent messages and decodes images from them
        :type predictor: nn.Module
        :param quantizer: Quantization module that maps continuous tokens onto the discrete vocabulary
        :type quantizer: nn.Module
        :param vae: Autoencoder used for mapping images into compressed latent space where COMiT operates
        :type vae: nn.Module | None
        """
        super().__init__()

        self.quantizer = quantizer
        self.quantizer.dtype = torch.float32
        self.predictor = predictor
        self.vae = vae

        self.num_msg_tokens = self.predictor.num_msg_tokens
        self.msg_tokens_dim = self.predictor.msg_tokens_dim

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        """
        Map images into the compressed latent space of the VAE

        :param x: Batch of images of shape [b C H W]
        :type x: Tensor
        :return: Batch of latents of shape [b c h w]
        :rtype: Tensor
        """
        if self.vae is not None:
            x = self.vae.encode(x)

        return x

    @torch.no_grad()
    def decode(self, z: Tensor) -> Tensor:
        """
        Decodes from the VAE's latent space to pixels

        :param z: Batch of latents of shape [b c h w]
        :type z: Tensor
        :return: Batch of images of shape [b C H W]
        :rtype: Tensor
        """

        if self.vae is not None:
            z = self.vae.decode(z)

        return z

    def sample_prior_messages(self, b: int, device: torch.device) -> Tensor:
        """
        Initialize latent messages with a special token from the vocabulary

        :param self: Description
        :param b: Batch size
        :type b: int
        :param device: Device to initialize the latent messages on
        :type device: torch.device
        :return: The initialized latent messages of shape [b n d]
        :rtype: Tensor
        """
        msg_tokens_indices = torch.full(
            fill_value=self.quantizer.n_e // 2,
            size=[b, self.num_msg_tokens],
            dtype=torch.long,
            device=device,
        )
        msg_tokens = self.quantizer.embedding(msg_tokens_indices).detach()

        return msg_tokens

    def quantize(self, msg_tokens: Tensor, quantize: bool = False) -> Tensor:
        """
        Map continuous tokens onto the discrete vocabulary

        :param msg_tokens: Latent messages of shape [b n d]
        :type msg_tokens: Tensor
        :param quantize: Whether to apply quantization or just bound the tokens
        :type quantize: bool
        :return: The quantized tokens of shape [b n d]
        :rtype: Tensor
        """
        msg_tokens, _, _ = self.quantizer(msg_tokens, quantize=quantize)

        return msg_tokens

    def predict(
        self,
        t: Tensor,
        noisy_x_latents: Tensor,
        msg_tokens: Tensor,
        offsets: Tensor,
        quantize: bool = False,
        return_attn_weights: bool = False,
        attn_weights_layer: int | None = None,
    ) -> dict[str, Tensor]:
        """
        Forward pass through the predictor network that updates the latent messages and estimates the flow velocity
        field at the image latents

        :param t: Flow matching timestamp in [0, 1] as a tensor with shape [] (single element) or [b] (batch)
        :type t: Tensor
        :param noisy_x_latents: Noisy image latents of shape [b c h w]
        :type noisy_x_latents: Tensor
        :param msg_tokens: Current latent messages of shape [b n d]
        :type msg_tokens: Tensor
        :param offsets: Current offsets with respect to the last observed crop, tensor of shape [b 2]
        :type offsets: Tensor
        :param quantize: Whether to apply quantization to the updated messages
        :type quantize: bool
        :param return_attn_weights: Whether to return the attention maps
        :type return_attn_weights: bool
        :param attn_weights_layer: From which layer of the DiT to extract the attention maps
        :type attn_weights_layer: int | None
        :return: A dictionary containing the estimated velocities of shape [b c h w], the updates messages
        of shape [b n d] and optionally the attention maps of shape [b heads N N]
        :rtype: dict[str, Tensor]
        """
        # Prepare the timestamps
        if t.ndim < 1:
            t = torch.full([noisy_x_latents.size(0)], t, device=noisy_x_latents.device)

        # Obtain predictions
        predictor_results = self.predictor(
            noisy_x_latents,
            msg_tokens,
            t,
            offsets,
            None,
            return_attn_weights=return_attn_weights,
            attn_weights_layer=attn_weights_layer,
        )
        x_vectors = predictor_results[0]
        msg_updates = predictor_results[1]
        if return_attn_weights:
            attn_weights = predictor_results[-1]

        # Quantize and update the messages
        msg_tokens = self.quantize(msg_tokens=msg_updates, quantize=quantize)

        # Prepare the output dict
        return_dict = {
            "x_vectors": x_vectors,
            "msg_tokens": msg_tokens,
        }
        if return_attn_weights:
            return_dict["attn_weights"] = attn_weights

        return return_dict

    @torch.no_grad()
    def filter_crops(
        self,
        local_crops: list[Tensor],
        local_locations: list[Tensor],
        global_crop: bool = False,
        order: str | list[int] | list[list[int]] = "raster_scan",
        num_crops: int | None = None,
    ) -> tuple[list[Tensor], list[Tensor], list[int] | list[list[int]]]:
        """
        Reorder and optionally truncate the list of local crops

        :param local_crops: A list containing the local crops of shape [b c h w]
        :type local_crops: list[Tensor]
        :param local_locations: A list containing the locations of the crops of shape [b 2] each
        :type local_locations: list[Tensor]
        :param global_crop: Whether the first crop in the extracted sequence (before reordering) should be the global
        crop
        :type global_crop: bool
        :param order: One of ["raster_scan", "random", "adaptive"] or an explicit list of crop indices. If a list is
        provided, it should be either a list of indices (the same for all images in the batch) or a list of lists,
        where the [i, j] element corresponds to the index of the ith crop for the jth image in the batch. If global
        crop is required its index is 0 and the local crops have indices from 1 to 9 in the raster scan order.
        Otherwise, the local crops have indices from 0 to 8 in the raster scan order.
        :type order: str | list[int] | list[list[int]]
        :param num_crops: The number of crops to truncate the list of crops to (including the global crop if used.
        Applied after reordering.
        :type num_crops: int | None
        :return: The updated lists of local crops and their respecitve locations, and the explicit order as a list of
        crop indices (if the input order was batched, the output order will also be batched)
        :rtype: tuple[list[Tensor], list[Tensor], list[int] | list[list[int]]]
        """

        if num_crops is None:
            num_crops = len(local_crops)

        if not isinstance(order, list):
            if order == "random":
                if not global_crop:
                    order = torch.randperm(len(local_crops))
                else:
                    order = torch.randperm(len(local_crops) - 1) + 1
                    order = torch.cat([torch.tensor([0]), order])

            elif order == "raster_scan":
                order = list(range(len(local_crops)))

        if not isinstance(order[0], list):
            local_crops_reordered = [local_crops[i] for i in order]
            local_locations_reordered = [local_locations[i] for i in order]
        else:
            local_crops_reordered = []
            local_locations_reordered = []
            for order_elem in order:
                local_crops_reordered.append(
                    torch.stack([local_crops[i][bi] for bi, i in enumerate(order_elem)], dim=0)
                )
                local_locations_reordered.append(
                    torch.stack(
                        [local_locations[i][bi] for bi, i in enumerate(order_elem)],
                        dim=0,
                    )
                )

        local_crops = local_crops_reordered[:num_crops]
        local_locations = local_locations_reordered[:num_crops]
        order = order[:num_crops]

        return local_crops, local_locations, order

    @torch.no_grad()
    def prepare_crops(
        self,
        batch: dict[str, Any] | Tensor,
        global_crop: bool = False,
        order: str | list[int] | list[list[int]] = "raster_scan",
        num_crops: int | None = None,
        return_reconstructions: bool | None = False,
    ) -> dict[str, Tensor]:
        """
        Extract sequences of local crops according to the cropping policy specified with global_crop, order and
        num_crops.

        :param batch: Input batch of images or a dict containing previously extracted crops to filter (with keys
        "global_crops", "global_locations", "local_crops", "local_locations")
        :type batch: dict[str, Any] | Tensor
        :param global_crop: Whether the first crop in the extracted sequence (before reordering) should be the global
        crop
        :type global_crop: bool
        :param order: One of ["raster_scan", "random", "adaptive"] or an explicit list of crop indices. If a list is
        provided, it should be either a list of indices (the same for all images in the batch) or a list of lists,
        where the [i, j] element corresponds to the index of the ith crop for the jth image in the batch. If global
        crop is required its index is 0 and the local crops have indices from 1 to 9 in the raster scan order.
        Otherwise, the local crops have indices from 0 to 8 in the raster scan order.
        :type order: str | list[int] | list[list[int]]
        :param num_crops: The number of crops to truncate the list of crops to (including the global crop if used).
        Applied after reordering.
        :type num_crops: int | None
        :param return_reconstructions: Whether to return one-step reconstructions with sequentially updated messages
        (for non-dict batch only)
        :type return_reconstructions: bool | None
        :return: A dict containing "global_crops", "global_locations", "local_crops", "local_locations" and optionally
        one-step intermediate "reconstructions" and "crop_ids" for potential visualization
        :rtype: dict[str, Tensor]
        """

        if isinstance(batch, dict):
            assert "global_crops" in batch, "global_crops must be provided in the batch for dict input"
            assert "local_crops" in batch, "local_crops must be provided in the batch for dict input"
            assert "global_locations" in batch, "global_locations must be provided in the batch for dict input"
            assert "local_locations" in batch, "local_locations must be provided in the batch for dict input"

            global_crops = batch["global_crops"]
            local_crops = batch["local_crops"]
            global_locations = batch["global_locations"]
            local_locations = batch["local_locations"]

            local_crops, local_locations, _ = self.filter_crops(
                local_crops=local_crops,
                local_locations=local_locations,
                global_crop=global_crop,
                order=order,
                num_crops=num_crops,
            )

            return dict(
                global_crops=global_crops,
                local_crops=local_crops,
                global_locations=global_locations,
                local_locations=local_locations,
            )

        if isinstance(order, list) or order in ["raster_scan", "random"]:
            global_crops = batch
            global_locations = torch.zeros(global_crops.size(0), 2, device=global_crops.device)
            local_crops = []
            local_locations = []

            gs = (3, 3)

            for i in range(gs[1]):
                for j in range(gs[0]):
                    l_size = 96
                    g_size = global_crops.size(2)
                    top = i * ((g_size - l_size) // (gs[1] - 1))
                    left = j * ((g_size - l_size) // (gs[0] - 1))

                    crop = global_crops[:, :, top : top + l_size, left : left + l_size]
                    loc = torch.tensor([(left + l_size / 2) / 128 - 1, (top + l_size / 2) / 128 - 1])
                    loc = loc.unsqueeze(0).repeat(global_crops.size(0), 1).to(global_crops.device)  # [b 2]

                    local_crops.append(crop)
                    local_locations.append(loc)

            if global_crop:
                local_crops = [global_crops] + local_crops
                local_locations = [torch.zeros(global_crops.size(0), 2, device=global_crops.device)] + local_locations

            local_crops, local_locations, order = self.filter_crops(
                local_crops=local_crops,
                local_locations=local_locations,
                global_crop=global_crop,
                order=order,
                num_crops=num_crops,
            )

            return_dict = dict(
                global_crops=global_crops,
                local_crops=local_crops,
                global_locations=global_locations,
                local_locations=local_locations,
            )

            if return_reconstructions is not None and return_reconstructions:
                recs = []
                for i in range(len(local_crops)):
                    rec = self.reconstruct(
                        batch=return_dict,
                        num_crops=i + 1,
                        num_steps=1,
                        odesolver="euler",
                        cfg_weight=1.0,
                        apg_momentum=-0.5,
                        decode=True,
                    )["generated"]
                    recs.append(rec)

                return_dict["reconstructions"] = recs
                return_dict["crop_ids"] = order

            return return_dict

        elif order == "adaptive":
            assert num_crops is not None, "num_crops must be specified for the adaptive policy"

            global_crops = batch
            # Start with center crop if global_crop else start with empty
            crops = [([0] * global_crops.size(0) if global_crop else [4] * global_crops.size(0))]
            forbidden_mask = torch.zeros((1, 3, 3), device=global_crops.device)
            if not global_crop:
                forbidden_mask[0, 1, 1] = 1  # Center crop is already given
            forbidden_mask = forbidden_mask.expand(global_crops.size(0), -1, -1)  # [b 3 3]
            recs = []
            for i in range(num_crops):
                rec = self.reconstruct(
                    batch=global_crops,
                    num_crops=i + 1,
                    global_crop=global_crop,
                    order=crops,
                    num_steps=1,
                    odesolver="euler",
                    cfg_weight=1.0,
                    apg_momentum=-0.5,
                    decode=True,
                )["generated"]
                recs.append(rec)

                if i < num_crops - 1:
                    scores = (rec - global_crops).pow(2).sum(1)  # [b h w]
                    scores = F.avg_pool2d(scores, kernel_size=96, stride=80)  # [b 3 3]
                    scores = torch.where(
                        forbidden_mask == 1,
                        torch.tensor(float("-inf"), device=global_crops.device),
                        scores,
                    )
                    next_crops = torch.argmax(scores.view(global_crops.size(0), -1), dim=1)  # [b]
                    forbidden_mask[
                        torch.arange(global_crops.size(0)),
                        next_crops.to(global_crops.device) // 3,
                        next_crops.to(global_crops.device) % 3,
                    ] = 1
                    if global_crop:
                        next_crops += 1
                    crops.append(next_crops.tolist())

            return_dict = self.prepare_crops(
                batch=global_crops,
                global_crop=global_crop,
                order=crops,
                num_crops=num_crops,
            )

            if return_reconstructions:
                return_dict["reconstructions"] = recs
                return_dict["crop_ids"] = crops

            return return_dict

        else:
            raise NotImplementedError(
                "Only explicit orders or orders in " "['raster_scan', 'random', 'adaptive'] are implemented"
            )

    def tokenize(
        self,
        batch: Tensor | dict[str, Tensor],
        global_crop: bool = False,
        order: str | list[int] = "raster_scan",
        num_crops: int | None = None,
    ) -> dict[str, Tensor]:
        """
        Tokenize images into latent messages of discrete tokens

        :param batch: Input batch of images of shape [b c h w] or a dict containing previously extracted crops to
        filter (with keys "global_crops", "global_locations", "local_crops", "local_locations")
        :type batch: Tensor | dict[str, Tensor]
        :param global_crop: Whether the first crop in the extracted sequence (before reordering) should be the global
        crop
        :type global_crop: bool
        :param order: One of ["raster_scan", "random", "adaptive"] or an explicit list of crop indices. If a list is
        provided, it should be either a list of indices (the same for all images in the batch) or a list of lists,
        where the [i, j] element corresponds to the index of the ith crop for the jth image in the batch. If global
        crop is required its index is 0 and the local crops have indices from 1 to 9 in the raster scan order.
        Otherwise, the local crops have indices from 0 to 8 in the raster scan order.
        :type order: str | list[int] | list[list[int]]
        :param num_crops: The number of crops to truncate the list of crops to (including the global crop if used).
        Applied after reordering.
        :type num_crops: int | None
        :return: A dict containing the latent messages of shape [b n d] and the offsets to use for decoding, tensor of
        shape [b 2]
        :rtype: dict[str, Tensor]
        """

        # ------------- Prepare crops -------------
        crops_dict = self.prepare_crops(
            batch=batch,
            global_crop=global_crop,
            order=order,
            num_crops=num_crops,
            return_reconstructions=False,
        )
        global_crops = crops_dict["global_crops"]
        local_crops = crops_dict["local_crops"]
        global_locations = crops_dict["global_locations"]
        local_locations = crops_dict["local_locations"]
        num_crops = len(local_crops)

        # ------------- Derive msgs from crops -------------

        local_locations = torch.stack(local_locations, dim=1)  # [b num_local_crops 2]
        offsets = torch.cat(
            [
                torch.zeros_like(local_locations[:, :1]),
                local_locations[:, 1:] - local_locations[:, :-1],
            ],
            dim=1,
        )  # [b num_local_crops 2]

        msgs = self.sample_prior_messages(b=global_crops.size(0), device=global_crops.device)  # [b n d]
        for i in range(num_crops):
            crop = local_crops[i]  # [b c h w]
            cur_offsets = offsets[:, i]  # [b 2]

            crop_latents = self.encode(crop)  # [b c h w]
            t = torch.ones(crop_latents.size(0), device=crop_latents.device)  # [b]

            pred_dict = self.predict(
                t=t,
                noisy_x_latents=crop_latents,
                msg_tokens=msgs,
                offsets=cur_offsets,
                quantize=True,
            )
            msgs = pred_dict["msg_tokens"]

        offsets = global_locations - local_locations[:, -1]  # [b 2]

        return dict(msgs=msgs, offsets=offsets)

    def detokenize(
        self,
        msgs: Tensor,
        offsets: Tensor,
        num_steps: int = 10,
        odesolver: str = "euler",
        cfg_weight: float = 6.0,
        apg_momentum: float = -0.5,
        return_attn_weights: bool = False,
        attn_weights_layer: int | None = None,
        attn_weights_time: float | None = None,
        decode: bool = True,
    ) -> dict[str, Tensor]:
        """
        Decode the latent messages back to images.

        :param msgs: The latent messages of discrete tokens of shape [b n d]
        :type msgs: Tensor
        :param offsets: The offsets with respect to the last embedded crop of shape [b 2]
        :type offsets: Tensor
        :param num_steps: Number of ODE discretization steps
        :type num_steps: int
        :param odesolver: The ODE solver to use for numercal integration of the velocity field (e.g. "euler" or
        "midpoint")
        :type odesolver: str
        :param cfg_weight: The strangth of the classifier-free guidance
        :type cfg_weight: float
        :param apg_momentum: The adaptive projected guidance momentum
        :type apg_momentum: float
        :param return_attn_weights: Whether to return the attention maps
        :type return_attn_weights: bool
        :param attn_weights_layer: The layer of the DiT to extract the attention maps from
        :type attn_weights_layer: int | None
        :param attn_weights_time: The denoising timestamp to extract the attention maps for. Must be > 1 / num_steps
        :type attn_weights_time: float | None
        :param decode: Whether to decode the resulting VAE latents back to pixels
        :type decode: bool
        :return: A dict containing the generated images of shape [b C H W] in "generated" and the attention maps of
        shape [b heads N N] in "attn_weights"
        :rtype: dict[str, Tensor]
        """

        if return_attn_weights:
            assert attn_weights_layer is not None, "attn_weights_layer must be provided if return_attn_weights is True"
            assert attn_weights_time is not None, "attn_weights_time must be provided if return_attn_weights is True"

        # ------------- Prepare inputs -------------

        b = msgs.size(0)
        ch = self.predictor.in_channels
        h = self.predictor.max_input_size
        w = self.predictor.max_input_size
        device = msgs.device

        # Empty msgs
        msgs_empty = self.sample_prior_messages(b=b, device=device)  # [b n d]

        # ------------- Sample from ODE -------------

        momentum_buffer = MomentumBuffer(momentum=apg_momentum) if cfg_weight > 0.0 else None
        ode_state = {"attn_weights": None}

        def f(t, x):
            predict_dict = self.predict(
                t=t,
                noisy_x_latents=x,
                msg_tokens=msgs,
                offsets=offsets,
                return_attn_weights=return_attn_weights,
                attn_weights_layer=attn_weights_layer,
            )
            x_vec = predict_dict["x_vectors"]
            if return_attn_weights and ode_state["attn_weights"] is None and t >= attn_weights_time:
                ode_state["attn_weights"] = predict_dict["attn_weights"]
            if cfg_weight == 0.0:
                return x_vec
            x_vec_unc = self.predict(t=t, noisy_x_latents=x, msg_tokens=msgs_empty, offsets=offsets)["x_vectors"]
            return adaptive_projected_guidance(
                x_vec,
                x_vec_unc,
                guidance_scale=cfg_weight,
                momentum_buffer=momentum_buffer,
                eta=0.0,
                norm_threshold=2.5,
            )

        t = torch.linspace(0.0, 1.0, steps=num_steps + 1, device=device)
        y = torch.randn([b, ch, h, w], device=device)
        x = odeint(
            f,
            y,
            t,
            method=odesolver,
            atol=1e-5,
            rtol=1e-5,
            adjoint_params=self.predictor.parameters(),
        )[-1]

        # ------------- Decode generated -------------
        if decode:
            x = self.decode(x)

        generated = x

        return dict(
            generated=generated,
            attn_weights=ode_state["attn_weights"],
        )

    def reconstruct(
        self,
        batch: Tensor,
        global_crop: bool = False,
        order: str | list[int] = "raster_scan",
        num_crops: int | None = None,
        num_steps: int = 134,
        odesolver: str = "euler",
        cfg_weight: float = 0.0,
        apg_momentum: float = 0.0,
        return_attn_weights: bool = False,
        attn_weights_layer: int | None = None,
        attn_weights_time: float | None = None,
        decode: bool = True,
    ) -> dict[str, Tensor]:
        """
        A pipeline to tokenize and detokenize images

        :param batch: Input batch of images of shape [b c h w] or a dict containing previously extracted crops to
        filter (with keys "global_crops", "global_locations", "local_crops", "local_locations")
        :type batch: Tensor | dict[str, Tensor]
        :param global_crop: Whether the first crop in the extracted sequence (before reordering) should be the global
        crop
        :type global_crop: bool
        :param order: One of ["raster_scan", "random", "adaptive"] or an explicit list of crop indices. If a list is
        provided, it should be either a list of indices (the same for all images in the batch) or a list of lists,
        where the [i, j] element corresponds to the index of the ith crop for the jth image in the batch. If global
        crop is required its index is 0 and the local crops have indices from 1 to 9 in the raster scan order.
        Otherwise, the local crops have indices from 0 to 8 in the raster scan order.
        :type order: str | list[int] | list[list[int]]
        :param num_crops: The number of crops to truncate the list of crops to (including the global crop if used).
        Applied after reordering.
        :param num_steps: Number of ODE discretization steps
        :type num_steps: int
        :param odesolver: The ODE solver to use for numercal integration of the velocity field (e.g. "euler" or
        "midpoint")
        :type odesolver: str
        :param cfg_weight: The strangth of the classifier-free guidance
        :type cfg_weight: float
        :param apg_momentum: The adaptive projected guidance momentum
        :type apg_momentum: float
        :param return_attn_weights: Whether to return the attention maps
        :type return_attn_weights: bool
        :param attn_weights_layer: The layer of the DiT to extract the attention maps from
        :type attn_weights_layer: int | None
        :param attn_weights_time: The denoising timestamp to extract the attention maps for. Must be > 1 / num_steps.
        :type attn_weights_time: float | None
        :param decode: Whether to decode the resulting VAE latents back to pixels
        :type decode: bool
        :return: A dict containing the generated images of shape [b C H W] in "generated" and the attention maps of
        shape [b heads N N] in "attn_weights"
        :rtype: dict[str, Tensor]
        """

        # ------------- Tokenize -------------

        tokenize_dict = self.tokenize(batch=batch, global_crop=global_crop, order=order, num_crops=num_crops)
        msgs = tokenize_dict["msgs"]
        offsets = tokenize_dict["offsets"]

        # ------------- Detokenize -------------

        detokenize_dict = self.detokenize(
            msgs=msgs,
            offsets=offsets,
            num_steps=num_steps,
            odesolver=odesolver,
            cfg_weight=cfg_weight,
            apg_momentum=apg_momentum,
            return_attn_weights=return_attn_weights,
            attn_weights_layer=attn_weights_layer,
            attn_weights_time=attn_weights_time,
            decode=decode,
        )
        generated = detokenize_dict["generated"]
        attn_weights = detokenize_dict["attn_weights"]

        return dict(
            generated=generated,
            attn_weights=attn_weights,
            msgs=msgs,
            offsets=offsets,
        )


class COMiT(COMiTBase, PyTorchModelHubMixin):
    def __init__(self, config: dict[str, Any]):
        self.config = copy.deepcopy(config)

        super().__init__(
            predictor=instantiate(config["predictor"]),
            quantizer=instantiate(config["quantizer"]),
            vae=instantiate(config["vae"]),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        return cls(config)
