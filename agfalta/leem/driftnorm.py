"""
Tools for drift alignment and normalization.
"""
# pylint: disable=missing-docstring
# pylint: disable=invalid-name
# pylint: disable=no-member
# pylint: disable=c-extension-no-member

import cv2 as cv
import numpy as np

from agfalta.leem.utility import ProgressBar, try_load_img, try_load_stack



def normalize_image(img, mcp, dark_counts=100):
    img = try_load_img(img)
    mcp = try_load_img(mcp)
    if not isinstance(dark_counts, (int, float, complex)):
        dark_image = try_load_img(dark_counts)
        dark_counts = dark_image.data

    img = img.copy()
    try:
        normed_mcp = np.clip(np.nan_to_num(mcp.data - dark_counts), 1, None)
        normed = (img.data - dark_counts) / normed_mcp
    except ValueError:
        raise ValueError("Normalize: Resolution of MCP or dark image does not match.")
    img.data = np.nan_to_num(np.clip(normed, 0, None))
    return img

def normalize_stack(stack, mcp, dark_counts=100):
    progbar = ProgressBar(len(stack), suffix="Normalizing...")
    stack = try_load_stack(stack)
    mcp = try_load_img(mcp)
    if not isinstance(dark_counts, (int, float, complex)):
        dark_counts = try_load_img(dark_counts)

    stack_normed = stack.copy()
    for i, img in enumerate(stack):
        stack_normed[i] = normalize_image(img, mcp, dark_counts=dark_counts)
        progbar.increment()
    progbar.finish()
    # is this monkey-patching necessary?:
    stack_normed.mcp = mcp
    stack_normed.dark_counts = dark_counts
    return stack_normed

def align_stack(stack, **kwargs):
    stack = try_load_stack(stack)
    alignment = find_alignment_matrices(stack, **kwargs)
    stack = apply_alignment_matrices(stack, alignment)
    stack.alignment = alignment
    return stack

def apply_alignment_matrices(stack, alignment):
    stack = stack.copy()
    for warp_matrix, img in zip(alignment, stack[1:]):
        img.data = cv.warpPerspective(
            img.data, warp_matrix, img.data.shape[::-1],
            flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP
        )
    return stack

def find_alignment_matrices(stack, algorithm="sift", **kwargs):
    if algorithm == "ecc":
        return find_alignment_matrices_ecc(stack, **kwargs)
    elif algorithm == "sift":
        return find_alignment_matrices_sift(stack, **kwargs)
    raise ValueError(f"Unknown algorithm '{algorithm}'")


def find_alignment_matrices_sift(stack, trafo="full-affine", min_matches=10, mask_outer=0.2,
                                 **_kwargs):
    """Trafo can either be "full-affine", "affine"(=rigid) or "homography"."""
    # pylint: disable=too-many-locals
    sift = cv.SIFT_create()
    data8bit = []
    # cut off outer fraction of image
    dy, dx = np.array(mask_outer * np.array(stack[0].data.shape), dtype=np.int)
    progbar = ProgressBar(len(stack) * 2 - 1, suffix="Calculating drift (SIFT)...")
    for img in stack:
        # sift needs 8-bit images:
        img8bit = cv.normalize(
            img.data[dy:-dy, dx:-dx],
            None, 0, 255, cv.NORM_MINMAX
        ).astype("uint8")
        # find keypoints and descriptors:
        kp, desc = sift.detectAndCompute(img8bit, None)
        data8bit.append((img8bit, kp, desc))
        progbar.increment()

    alignment = [np.eye(3, 3, dtype=np.float32)]
    # bf = cv.BFMatcher()#cv.NORM_L1)
    bf = cv.FlannBasedMatcher()
    for i in range(len(data8bit) - 1):
        # match the descriptors. each match object consists of two best matches
        matches = bf.knnMatch(data8bit[i][2], data8bit[i + 1][2], 2)
        good_matches = []
        for m, n in matches:
            # if the two best matches are too similar, the match is discarded
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) < min_matches:
            if "failed at image" not in progbar.suffix:
                progbar.suffix += f" | failed at image"
            progbar.suffix += f" {i}"
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            src = np.array([data8bit[i][1][m.queryIdx].pt for m in good_matches],
                           dtype=np.float32).reshape(-1, 1, 2)
            dst = np.array([data8bit[i + 1][1][m.trainIdx].pt for m in good_matches],
                           dtype=np.float32).reshape(-1, 1, 2)
            if trafo == "homography":
                warp_matrix, _ = cv.findHomography(src, dst)
            else:
                warp_matrix = cv.estimateRigidTransform(src, dst, trafo == "full-affine")
                if trafo == "translation":
                    warp_matrix[:2, :2] = [[1, 0], [0, 1]]
                warp_matrix = np.append(warp_matrix, [[0, 0, 1]], axis=0)
        warp_matrix = np.dot(alignment[-1], warp_matrix)
        alignment.append(warp_matrix)
        progbar.increment()
    progbar.finish()
    return alignment


def find_alignment_matrices_ecc(stack, max_iter=500, eps=1e-10, mask_outer=0.2, **_kwargs):
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, max_iter, eps)
    warp_mode = cv.MOTION_TRANSLATION
    mask = np.ones_like(stack[0].data, dtype=np.uint8)
    # cut off outer fraction of image
    cutoff_y, cutoff_x = np.array(mask_outer * np.array(mask.shape), dtype=np.int)
    # pylint: disable=unsupported-assignment-operation
    mask[cutoff_y:-cutoff_y, :] = 0
    mask[:, cutoff_x:-cutoff_x] = 0

    alignment = [np.eye(3, 3, dtype=np.float32)]

    progbar = ProgressBar(len(stack), suffix="Calculating drift (ECC)...")
    for i in range(len(stack) - 1):
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        _, warp_matrix = cv.findTransformECC(
            stack[i].data, stack[i + 1].data,
            warp_matrix, warp_mode, criteria,
            mask,   # all ones, so no mask at all
            5       # gaussian blur to apply before
        )
        # warp_matrix looks like this: [[1, 0, dx], [0, 1, dy]].
        # Add another [[0, 0, 1]] to get a 3x3 transformation matrix
        warp_matrix = np.append(warp_matrix, [[0, 0, 1]], axis=0)
        warp_matrix = np.dot(alignment[-1], warp_matrix)
        alignment.append(warp_matrix)
        progbar.increment()
    progbar.finish()
    return alignment
