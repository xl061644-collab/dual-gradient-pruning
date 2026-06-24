import torch
from functools import partial
import warnings
import logging

log = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore", message=".*torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.*",
)  # this is kornia's problem

import torch
#
#
# from .metrics import psnr_compute, registered_psnr_compute

# Block ImageNet corrupt EXIF warnings
import warnings

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def psnr_compute(img_batch, ref_batch, batched=False, factor=1.0, clip=False):
    """Standard PSNR."""
    if clip:
        img_batch = torch.clamp(img_batch, 0, 1)

    if batched:
        mse = ((img_batch.detach() - ref_batch) ** 2).mean()
        if mse > 0 and torch.isfinite(mse):
            return 10 * torch.log10(factor ** 2 / mse)
        elif not torch.isfinite(mse):
            return [torch.tensor(float("nan"), device=img_batch.device)] * 2
        else:
            return [torch.tensor(float("inf"), device=img_batch.device)] * 2
    else:
        B = img_batch.shape[0]
        mse_per_example = ((img_batch.detach() - ref_batch) ** 2).view(B, -1).mean(dim=1)
        if any(mse_per_example == 0):
            return [torch.tensor(float("inf"), device=img_batch.device)] * 2
        elif not all(torch.isfinite(mse_per_example)):
            return [torch.tensor(float("nan"), device=img_batch.device)] * 2
        else:
            psnr_per_example = 10 * torch.log10(factor ** 2 / mse_per_example)
            return psnr_per_example.mean().item(), psnr_per_example.max().item()


def registered_psnr_compute(img_batch, ref_batch, factor=1.0):
    """Use kornia for now."""
    return _registered_psnr_compute_kornia(img_batch, ref_batch, factor)


def _registered_psnr_compute_kornia(img_batch, ref_batch, factor=1.0):
    """Kornia version. Todo: Use a smarter/deeper matching tool."""
    try:
        from kornia.geometry import ImageRegistrator, HomographyWarper  # lazy import here as well
    except ModuleNotFoundError:
        warnings.warn("To utilize registered PSNR, install kornia.")
        return torch.as_tensor(float("NaN")), torch.as_tensor(float("NaN"))

    B = img_batch.shape[0]
    default_psnrs = []
    registered_psnrs = []
    # If only this was parallelized, todo ...
    for img, ref in zip(img_batch.detach(), ref_batch.detach()):
        img, ref = img[None, ...], ref[None, ...]
        mse = ((img - ref) ** 2).mean()
        default_psnrs += [10 * torch.log10(factor ** 2 / mse)]
        # Align by homography:
        registrator = ImageRegistrator("similarity", num_iterations=2500)
        registrator.warper = partial(HomographyWarper, padding_mode="reflection")
        registrator.to(ref.device)
        homography = registrator.register(img, ref)
        warped_img = registrator.warp_src_into_dst(img)
        # Compute new PSNR:
        mse = ((warped_img.detach() - ref_batch) ** 2).mean()
        registered_psnrs += [10 * torch.log10(factor ** 2 / mse)]

    # Return best of default and warped PSNR:
    result = torch.stack([torch.stack(default_psnrs), torch.stack(registered_psnrs)]).max(dim=0)[0]
    return result.mean().item(), result.max().item()


def cw_ssim(img_batch, ref_batch, scales=5, skip_scales=None, K=1e-6, reduction="mean"):
    """Batched complex wavelet structural similarity.

    As in Zhou Wang and Eero P. Simoncelli, "TRANSLATION INSENSITIVE IMAGE SIMILARITY IN COMPLEX WAVELET DOMAIN"
    Ok, not quite, this implementation computes no local SSIM and neither averaging over local patches and uses only
    the existing wavelet structure to provide a similar scale-invariant decomposition.

    skip_scales can be a list like [True, False, False, False] marking levels to be skipped.
    K is a small fudge factor.
    """
    try:
        from pytorch_wavelets import DTCWTForward
    except ModuleNotFoundError:
        warnings.warn(
            "To utilize wavelet SSIM, install pytorch wavelets from https://github.com/fbcotter/pytorch_wavelets."
        )
        return torch.as_tensor(float("NaN")), torch.as_tensor(float("NaN"))

    # 1) Compute wavelets:
    setup = dict(device=img_batch.device, dtype=img_batch.dtype)
    if skip_scales is not None:
        include_scale = [~s for s in skip_scales]
        total_scales = scales - sum(skip_scales)
    else:
        include_scale = True
        total_scales = scales
    xfm = DTCWTForward(J=scales, biort="near_sym_b", qshift="qshift_b", include_scale=include_scale).to(**setup)
    # print("xfm", xfm)
    # print("img_batch",img_batch)
    img_coefficients = xfm(img_batch)
    ref_coefficients = xfm(ref_batch)

    # 2) Multiscale complex SSIM:
    ssim = 0
    for xs, ys in zip(img_coefficients[1], ref_coefficients[1]):
        if len(xs) > 0:
            xc = torch.view_as_complex(xs)
            yc = torch.view_as_complex(ys)

            conj_product = (xc * yc.conj()).sum(dim=2).abs()
            square_img = (xc * xc.conj()).abs().sum(dim=2)
            square_ref = (yc * yc.conj()).abs().sum(dim=2)

            ssim_val = (2 * conj_product + K) / (square_img + square_ref + K)
            ssim += ssim_val.mean(dim=[1, 2, 3])
    ssim = ssim / total_scales
    return ssim.mean().item(), ssim.max().item()


def psnr_compute(img_batch, ref_batch, batched=False, factor=1.0, clip=False):
    """Standard PSNR."""
    if clip:
        img_batch = torch.clamp(img_batch, 0, 1)

    if batched:
        mse = ((img_batch.detach() - ref_batch) ** 2).mean()
        if mse > 0 and torch.isfinite(mse):
            return 10 * torch.log10(factor ** 2 / mse)
        elif not torch.isfinite(mse):
            return [torch.tensor(float("nan"), device=img_batch.device)] * 2
        else:
            return [torch.tensor(float("inf"), device=img_batch.device)] * 2
    else:
        B = img_batch.shape[0]
        mse_per_example = ((img_batch.detach() - ref_batch) ** 2).view(B, -1).mean(dim=1)
        if any(mse_per_example == 0):
            return [torch.tensor(float("inf"), device=img_batch.device)] * 2
        elif not all(torch.isfinite(mse_per_example)):
            return [torch.tensor(float("nan"), device=img_batch.device)] * 2
        else:
            psnr_per_example = 10 * torch.log10(factor ** 2 / mse_per_example)
            return psnr_per_example.mean().item(), psnr_per_example.max().item()





def _run_vision_metrics1(
        reconstructed_user_data,
        true_user_data,
        compute_rpsnr=True,
        setup=dict(device=torch.device("cpu"), dtype=torch.float),
        # mean=[0.485, 0.456, 0.406],
        # std=[0.229, 0.224, 0.225],
        mean=[0.5072, 0.4867, 0.4412],
        std=[0.2673, 0.2564, 0.2762]
):
    import lpips  # lazily import this only if vision reporting is used.

    lpips_scorer = lpips.LPIPS(net="alex", verbose=False).to(**setup)

    # metadata = server_payload[0]["metadata"]
    # print(metadata)
    # if hasattr(metadata, "mean"):
    #     print("**********************************")
    #
    # else:
    #     dm, ds = torch.tensor(0, **setup), torch.tensor(1, **setup)

    dm = torch.as_tensor(mean, **setup)[None, :, None, None]
    ds = torch.as_tensor(std, **setup)[None, :, None, None]

    # rec_denormalized = torch.clamp(reconstructed_user_data.to(**setup) * ds + dm, 0, 1)
    # ground_truth_denormalized = torch.clamp(true_user_data.to(**setup) * ds + dm, 0, 1)

    rec_denormalized = torch.clamp(reconstructed_user_data["data"].to(**setup) * ds + dm, 0, 1)
    ground_truth_denormalized = torch.clamp(true_user_data["data"].to(**setup) * ds + dm, 0, 1)


    # print("rec_denormalized ",rec_denormalized )

    mse_score = (rec_denormalized - ground_truth_denormalized).pow(2).mean(dim=[1, 2, 3])
    avg_mse, max_mse = mse_score.mean().item(), mse_score.max().item()
    avg_psnr, max_psnr = psnr_compute(rec_denormalized, ground_truth_denormalized, factor=1)
    avg_ssim, max_ssim = cw_ssim(rec_denormalized, ground_truth_denormalized, scales=5)

    # Hint: This part switches to the lpips [-1, 1] normalization:
    lpips_score = lpips_scorer(rec_denormalized, ground_truth_denormalized, normalize=True)
    avg_lpips, min_lpips = lpips_score.mean().item(), lpips_score.min().item()

    # print()

    # Compute registered psnr. This is a bit computationally intensive:
    if compute_rpsnr:
        avg_rpsnr, max_rpsnr = registered_psnr_compute(rec_denormalized, ground_truth_denormalized, factor=1)
    else:
        avg_rpsnr, max_rpsnr = float("nan"), float("nan")

    vision_metrics = dict(
        mse=avg_mse,
        psnr=avg_psnr,
        lpips=avg_lpips,
        rpsnr=avg_rpsnr,
        ssim=avg_ssim,
        max_ssim=max_ssim,
        min_lpips=min_lpips,
        max_rpsnr=max_rpsnr,
        # **{f"IIP-{k}": v for k, v in iip_scores.items()},
    )


    return vision_metrics
