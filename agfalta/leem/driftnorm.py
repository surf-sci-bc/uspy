"""
Tools for drift alignment and normalization.
"""
# pylint: disable=missing-docstring
# pylint: disable=invalid-name
# pylint: disable=no-member
# pylint: disable=c-extension-no-member

import cv2 as cv
import numpy as np

from agfalta.leem.utility import imgify, stackify
from agfalta.utility import progress_bar
from agfalta.leem.processing import ROI


def normalize(img_or_stack, *args, **kwargs):
    try:
        img = imgify(img_or_stack)
        return normalize_image(img, *args, **kwargs)
    except FileNotFoundError:
        stack = stackify(img_or_stack)
        return normalize_stack(stack, *args, **kwargs)


def normalize_image(img, mcp, dark_counts=100):
    img = imgify(img)
    mcp = imgify(mcp)
    if not isinstance(dark_counts, (int, float, complex)):
        dark_image = imgify(dark_counts)
        dark_counts = dark_image.data

    img = img.copy()
    try:
        normed_mcp = np.clip(np.nan_to_num(mcp.data - dark_counts), 1, None)
        normed = (img.data - dark_counts) / normed_mcp
    except ValueError as e:
        raise ValueError(
            "Normalize: Resolution of MCP or dark image does not match."
        ) from e
    img.data = np.nan_to_num(np.clip(normed, 0, None))
    return img


def normalize_stack(stack, mcp, dark_counts=100):
    stack = stackify(stack)
    mcp = imgify(mcp)
    if not isinstance(dark_counts, (int, float, complex)):
        dark_counts = imgify(dark_counts)

    stack_normed = stack.copy()
    for i, img in enumerate(progress_bar(stack, "Normalizing...")):
        stack_normed[i] = normalize_image(img, mcp, dark_counts=dark_counts)
    # is this monkey-patching necessary?:
    stack_normed.mcp = mcp
    stack_normed.dark_counts = dark_counts
    return stack_normed


def align(stack, **kwargs):
    """
    Use these keyword arguments:
    algorithm={"ecc"}
    roi: defines a ROI, which can be any shape (circle, rectangle...).
    Defaults to 15% rectangular cutoff
    for ecc:
        trafo={"translation","rigid","affine"}   (default=translation)
        max_iter=int        maximum iterations, default: 500
        eps=number          threshold to reach, default: 1e-4
        avg=int             alignment ist averaged by matching with avg Number of previous images,
                            default: 1
    """
    stack = stackify(stack)
    alignment = find_alignment_matrices(stack, **kwargs)
    stack = apply_alignment_matrices(stack, alignment)
    stack.alignment = alignment
    return stack, alignment


def align_stack(*args, **kwargs):
    print("align_stack() is DEPRECATED, use align()")
    return align(*args, **kwargs)


def apply_alignment_matrices(stack, alignment):
    stack = stack.copy()
    for warp_matrix, img in zip(alignment, stack):
        img.data = cv.warpPerspective(
            img.data,
            warp_matrix,
            img.data.shape[::-1],
            flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP,
        )
    return stack


def find_alignment_matrices(stack, algorithm="ecc", roi=None, **kwargs):

    if roi is None or not isinstance(roi, ROI):
        print("No valid ROI found. Creating default ROI with 15% cutoff on each side")
        y0, x0 = np.array(np.shape(stack[0].data), dtype=int) * 0.15
        height, width = np.array(np.shape(stack[0].data)) * 0.7
        roi = ROI(x0, y0, type_="rectangle", width=width, height=height)

    stack = stack.copy()
    img_height, img_width = np.shape(stack[0].data)
    mask = np.array(roi.create_mask(img_height, img_width), dtype=np.uint8)

    if algorithm == "ecc":
        return do_ecc_align(stack, mask=mask, **kwargs)

    raise ValueError(f"Unknown algorithm '{algorithm}'")


def do_ecc_align(
    stack, max_iter=500, eps=1e-4, trafo="translation", mask=None, avg=1, **_kwargs
):

    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, max_iter, eps)
    if trafo == "translation":
        warp_mode = cv.MOTION_TRANSLATION
    elif trafo in ("euclidean", "rigid"):
        warp_mode = cv.MOTION_EUCLIDEAN
    elif trafo == "affine":
        warp_mode = cv.MOTION_AFFINE
        #if avg != 1:
        #    raise ValueError("avg not supported for affine transformation")
    else:
        print("Unrecognized transformation. Using Translation.")
        warp_mode = cv.MOTION_TRANSLATION

    alignment = [np.eye(3, 3, dtype=np.float32)]
    # Matrix that holds all transformation matrices
    warp_matrices = np.zeros((len(stack), len(stack), 3, 3), dtype=np.float32)
    failed_aligns = []

    for ii in progress_bar(range(0, len(stack)), "Calculating drift (ECC)"):
        for jj in range(len(stack)):
            try:
                # We only care for upper part of matrix. W(ii,jj) = W(jj,ii)^-1
                if jj < ii:
                    continue
                if ii == jj:  # Images are alignt with themselves
                    warp_matrices[ii, jj] = np.eye(3, 3, dtype=np.float32)
                    continue
                if jj - ii > avg:  # only match next avg images
                    break

                warp_matrix = np.eye(2, 3, dtype=np.float32)

                _, warp_matrix = cv.findTransformECC(  # template = warp_matrix * input
                    stack[ii].data,  # ii is template image
                    stack[jj].data,  # jj is input image
                    warp_matrix,
                    warp_mode,
                    criteria,
                    mask,  # hide everythin that is not in ROI
                    5,  # gaussian blur to apply before
                )
                warp_matrix = np.append(
                    warp_matrix, [[0, 0, 1]], axis=0
                )  # Expand to 3x3 matrix
                warp_matrices[ii, jj, :, :] = warp_matrix
            except:
                print(f"ECC failed to match images {ii} and {jj}.")
                failed_aligns.append((ii,jj))
                continue

    for ii in range(1, len(stack)):
        shift = np.zeros((3, 3), dtype=np.float32)
        fail = 0
        for jj in range(max(0, ii - avg), ii):
            if (jj,ii) in failed_aligns:
                fail += 1
                continue
            shift += (
                warp_matrices[jj, ii] @ alignment[jj]
            )  # shift is calculated as average of drifts of previous avg images with current image
        if min(ii+1,avg) == fail:
            print(f"Alignment of Image {ii} ulimatly failed. Assuming linear drift.")
            shift = 2*alignment[-1]-alignment[-2]
        else:
            shift = shift / (min(ii + 1, avg)-fail)

        alignment.append(shift)

    return alignment
