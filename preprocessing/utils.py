from typing import List
import numpy as np
def get_bbox_from_mask(mask: np.ndarray) -> List[List[int]]:
    """
    ALL bounding boxes in acvl_utils and nnU-Netv2 are half open interval [start, end)!
    - Alignment with Python Slicing
    - Ease of Subdivision
    - Consistency in Multi-Dimensional Arrays
    - Precedent in Computer Graphics
    https://chatgpt.com/share/679203ec-3fbc-8013-a003-13a7adfb1e73

    this implementation uses less ram than the np.where one and is faster as well IF we expect the bounding box to
    be close to the image size. If it's not it's likely slower!

    :param mask:
    :param outside_value:
    :return:
    """
    Z, X, Y = mask.shape
    minzidx, maxzidx, minxidx, maxxidx, minyidx, maxyidx = 0, Z, 0, X, 0, Y
    zidx = list(range(Z))
    for z in zidx:
        if np.any(mask[z]):
            minzidx = z
            break
    for z in zidx[::-1]:
        if np.any(mask[z]):
            maxzidx = z + 1
            break

    xidx = list(range(X))
    for x in xidx:
        if np.any(mask[:, x]):
            minxidx = x
            break
    for x in xidx[::-1]:
        if np.any(mask[:, x]):
            maxxidx = x + 1
            break

    yidx = list(range(Y))
    for y in yidx:
        if np.any(mask[:, :, y]):
            minyidx = y
            break
    for y in yidx[::-1]:
        if np.any(mask[:, :, y]):
            maxyidx = y + 1
            break
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

def bounding_box_to_slice(bounding_box: List[List[int]]):
    """
    ALL bounding boxes in acvl_utils and nnU-Netv2 are half open interval [start, end)!
    - Alignment with Python Slicing
    - Ease of Subdivision
    - Consistency in Multi-Dimensional Arrays
    - Precedent in Computer Graphics
    https://chatgpt.com/share/679203ec-3fbc-8013-a003-13a7adfb1e73
    """
    return tuple([slice(*i) for i in bounding_box])

def resize_segmentation(segmentation, new_shape, order=3):
    '''
    Resizes a segmentation map. Supports all orders (see skimage documentation). Will transform segmentation map to one
    hot encoding which is resized and transformed back to a segmentation map.
    This prevents interpolation artifacts ([0, 0, 2] -> [0, 1, 2])
    :param segmentation:
    :param new_shape:
    :param order:
    :return:
    '''
    tpe = segmentation.dtype
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(segmentation.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False).astype(tpe)
    else:
        reshaped = np.zeros(new_shape, dtype=segmentation.dtype)

        unique_labels = np.sort(pd.unique(segmentation.ravel()))
        for i, c in enumerate(unique_labels):
            mask = segmentation == c
            reshaped_multihot = resize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
            reshaped[reshaped_multihot >= 0.5] = c
        return reshaped