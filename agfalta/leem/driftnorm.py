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
#from multiprocessing import Pool, cpu_count, Process, Manager


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
        raise ValueError("Normalize: Resolution of MCP or dark image does not match.") from e
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
    algorithm={"ecc","sift"}
    roi: defines a ROI, which can be any shape (circle, rectangle...). Defaults to 15% rectangular cutoff
    for ecc:
        trafo={"translation","rigid","affine"}   (default=translation)
        max_iter=int      (maximum iterations, default=500)
        eps=number        (threshold to reach, default=1e-10)
        avg=int           (alignment ist averaged by matching with avg Number of previous images, default = 1)
    for sift:
        trafo={"full-affine","affine","homography"}   (default=full-affine)
        min_matches=int    (minimum matches between two images, default=10)
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
    for warp_matrix, img in zip(alignment, stack[1:]):
        img.data = cv.warpPerspective(
            img.data, warp_matrix, img.data.shape[::-1],
            flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP
        )
    return stack

def find_alignment_matrices(stack, algorithm="sift", roi=None, **kwargs):
    if roi == None or not isinstance(roi, ROI):
        print("No valid ROI found. Creating default ROI with 15% cutoff on each side")
        y0, x0 = np.array(np.shape(stack[0].data), dtype=int)*0.15
        height, width = np.array(np.shape(stack[0].data))*0.7
        roi = ROI(x0, y0, type_="rectangle", width=width, height=height)

    img_height, img_width = np.shape(stack[0].data)
    mask = np.array(roi.create_mask(img_height, img_width), dtype=np.uint8)

    if algorithm == "ecc":
        return find_alignment_matrices_ecc(stack, mask=mask, **kwargs)
    elif algorithm == "sift":
        return find_alignment_matrices_sift(stack, mask=mask, **kwargs)
    raise ValueError(f"Unknown algorithm '{algorithm}'")


def find_alignment_matrices_sift(stack, trafo="full-affine", min_matches=10, mask=None,
                                 **_kwargs):
    """Trafo can either be "full-affine", "affine"(=rigid) or "homography"."""
    # pylint: disable=too-many-locals
    sift = cv.SIFT_create()
    data8bit = []
    for img in progress_bar(stack, "Finding keypoints (SIFT)..."):
        # sift needs 8-bit images:
        img8bit = cv.normalize(
            #img.data[dy:-dy, dx:-dx],
            img.data * mask,  # apply mask, so the ROI has enhanced contrast
            None, 0, 255, cv.NORM_MINMAX
        ).astype("uint8")

        # find keypoints and descriptors:
        kp, desc = sift.detectAndCompute(img8bit, mask)
        data8bit.append((img8bit, kp, desc))

    alignment = [np.eye(3, 3, dtype=np.float32)]
    # bf = cv.BFMatcher()#cv.NORM_L1)
    bf = cv.FlannBasedMatcher()
    for i in progress_bar(range(len(data8bit) - 1), "Matching keypoints..."):
        # match the descriptors. each match object consists of two best matches
        matches = bf.knnMatch(data8bit[i][2], data8bit[i + 1][2], 2)
        good_matches = []
        for m, n in matches:
            # if the two best matches are too similar, the match is discarded
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) < min_matches:
            print(f"No match at image {i}")
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            src = np.array(             # pylint: disable=too-many-function-args
                [data8bit[i][1][m.queryIdx].pt for m in good_matches],
                dtype=np.float32
            ).reshape(-1, 1, 2)
            dst = np.array(             # pylint: disable=too-many-function-args
                [data8bit[i + 1][1][m.trainIdx].pt for m in good_matches],
                dtype=np.float32
            ).reshape(-1, 1, 2)
            if trafo == "homography":
                warp_matrix, _ = cv.findHomography(src, dst)
            else:
                warp_matrix = cv.estimateRigidTransform(src, dst, trafo == "full-affine")
                if trafo == "translation":
                    warp_matrix[:2, :2] = [[1, 0], [0, 1]]
                warp_matrix = np.append(warp_matrix, [[0, 0, 1]], axis=0)
        warp_matrix = np.dot(alignment[-1], warp_matrix)
        alignment.append(warp_matrix)
    return alignment

def do_ecc(chunk):
    _, warp_matrix = cv.findTransformECC(*chunk)
    warp_matrix = np.append(warp_matrix, [[0, 0, 1]], axis=0)

    return warp_matrix

def find_alignment_matrices_ecc(stack, max_iter=500, eps=1e-10, trafo="translation", mask=None, avg=1, **_kwargs):

    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, max_iter, eps)
    if trafo == "translation":
        warp_mode = cv.MOTION_TRANSLATION
    elif trafo == "euclidean" or trafo == "rigid":
        warp_mode = cv.MOTION_EUCLIDEAN
    elif trafo == "affine":
        warp_mode = cv.MOTION_AFFINE
        if avg!=1:
            print("avg not supported for affine transformation")
    else:
        print("Unrecognized transformation. Using Translation.")
        warp_mode = cv.MOTION_TRANSLATION

    alignment = [np.eye(3, 3, dtype=np.float32)]

    #chunks = [ 1 for _ in range(len(stack)-1)]

    #for i, chunk in enumerate(chunks):
    #    warp_matrix = np.eye(2, 3, dtype=np.float32)
    #    chunks[i] = [stack[i].data, stack[i + 1].data, warp_matrix, warp_mode, criteria, mask, 5]
    
    #print(f"Calculating on {cpu_count()} cores")

    #manager = Manager()
    #return_dict = manager.dict()

    #processes = []
    #for chunk in chunks:
    #    p = Process(target=do_ecc, args=[chunk])
    #    processes.append(p)
    #    p.start()
    
    #for pro in processes:
    #    pro.join()

    #print(return_dict.values())

    #with Pool(1) as p:
    #    warp_matrix = p.map(do_ecc, chunks)

    #for warp_matrix in warp_matrix:
    #    warp_matrix = np.dot(alignment[-1], warp_matrix)
    #    alignment.append(warp_matrix)


    for i in progress_bar(range(len(stack) - 1), "Calculating drift (ECC)"):

        warp_matrix = [np.eye(2, 3, dtype=np.float32) for x in range(min(i+1,avg))]
        shift =  np.zeros((3,3))
        for j, matrix in enumerate(warp_matrix):
            _, matrix = cv.findTransformECC(
                stack[i-j].data, stack[i + 1].data,
                matrix, warp_mode, criteria,
                mask,   
                5       # gaussian blur to apply before
            )
            # warp_matrix looks like this: [[1, 0, dx], [0, 1, dy]].
            # Add another [[0, 0, 1]] to get a 3x3 transformation matrix
            matrix = np.append(matrix, [[0, 0, 1]], axis=0)
            shift += np.dot(alignment[-j-1], matrix)/len(warp_matrix)

        alignment.append(shift)
    

    return alignment