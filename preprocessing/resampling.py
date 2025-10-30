import time
import numpy as np
from collections import OrderedDict
from skimage.transform import resize
from copy import deepcopy
from scipy.ndimage import map_coordinates
import pandas as pd

from preprocessing.utils import resize_segmentation

class Resampler:
    """
    This class handle the resampling during the preprocessing and postprocessing
    """
    def __init__(self)-> None:
        """
        Initialize the resampler class
        """
        self.separate_z_anisotropy_threshold = 3.0
        self.force_separate_z = None
        self.is_seg=False
        self.order = 1
        self.order_z= 0

    def run(self,data : np.ndarray,current_spacing : tuple[float,float,float],new_spacing : tuple[float,float,float])->np.ndarray:
        """
        Resamples the input data to a new spacing

        Args:
            data (np.ndarray): Input array of shape (c, x, y, z)
            current_spacing (tuple[float,float,float]): Current spacing in mm (sx, sy, sz)
            new_spacing (tuple[float,float,float]): Desired spacing in mm (sx, sy, sz)

        Returns:
            np.ndarray: The resampled data 
        """
        do_separate_z, axis = self._determine_do_sep_z_and_axis(current_spacing, new_spacing,)

        if data is not None:
            assert data.ndim == 4, "data must be c x y z"

        shape = np.array(data.shape)
        new_shape = self._compute_new_shape(shape[1:], current_spacing, new_spacing)

        data_reshaped = self._resample_data_or_seg(data, new_shape,  axis, do_separate_z)
        return data_reshaped
    
    def _determine_do_sep_z_and_axis(self,current_spacing : tuple[float,float,float], new_spacing : tuple[float,float,float])->tuple[bool,int]:
        """
        @public
        Decides if resampling should be done separately along the z-axis and finds the right axis
        The choice is based on the force_separate_z attribute and the spacing values. If force_separate_z is set, it is used first. If not, the method checks if the current or new spacing needs separate resampling in the z direction

        Args:
            current_spacing (tuple[float,float,float]): Current spacing in mm (sx, sy, sz)
            new_spacing (tuple[float,float,float]): Desired spacing in mm (sx, sy, sz)

        Returns:
            tuple[bool,int]:
            - True if separate resampling along the z-axis should be done
            - The axis to resample separately, or None if not needed
        """
        if self.force_separate_z is not None:
            do_separate_z = self.force_separate_z
            if self.force_separate_z:
                axis = self._get_lowres_axis(current_spacing)
            else:
                axis = None
        else:
            if self._get_do_separate_z(current_spacing):
                do_separate_z = True
                axis = self._get_lowres_axis(current_spacing)
            elif self._get_do_separate_z(new_spacing):
                do_separate_z = True
                axis = self._get_lowres_axis(new_spacing)
            else:
                do_separate_z = False
                axis = None

        if axis is not None:
            if len(axis) == 3:
                do_separate_z = False
                axis = None
            elif len(axis) == 2:
                # this happens for spacings like (0.24, 1.25, 1.25) for example. In that case we do not want to resample
                # separately in the out of plane axis
                do_separate_z = False
                axis = None
            else:
                axis = axis[0]
        return do_separate_z, axis
    
    def _get_lowres_axis(self,new_spacing : tuple[float,float,float])->np.ndarray:
        """
        @public
        Finds which axis has the biggest spacing (=the lowest resolution)

        Args:
            new_spacing (tuple[float,float,float]): Spacing in mm (sx, sy, sz)

        Returns:
            np.ndarray: The indices of the axis with the biggest spacing
        """
        axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]  # find which axis is anisotropic
        return axis
    
    def _get_do_separate_z(self,spacing : tuple[float,float,float])->bool:
        """
        @public
         It compares the ratio between the max and min spacing values to a threshold (separate_z_anisotropy_threshold)

        Args:
            spacing (tuple[float,float,float]): Spacing in mm (sx, sy, sz)

        Returns:
            bool: True if the ratio is higher than the threshold
        """
        do_separate_z = (np.max(spacing) / np.min(spacing)) > self.separate_z_anisotropy_threshold
        return do_separate_z

    
    def _compute_new_shape(self, old_shape : np.ndarray, old_spacing : tuple[float,float,float], new_spacing : tuple[float,float,float]) -> np.ndarray:
        """
        @public
        Computes the new shape of data after resampling to a new spacing
        Scales each dimension according to the ratio of old and new spacing values

        Args:
            old_shape (np.ndarray): Current shape of the data
            old_spacing (tuple[float,float,float]): Current spacing in mm (sx, sy, sz) 
            new_spacing (tuple[float,float,float]): Desired spacing in mm (sx, sy, sz)

        Returns:
            np.ndarray: Future shape of the resempled data
        """
        assert len(old_spacing) == len(old_shape)
        assert len(old_shape) == len(new_spacing)
        new_shape = np.array([int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)])
        return new_shape
    
    def _resample_data_or_seg(self, data : np.ndarray, new_shape : np.ndarray, axis : int,do_separate_z : bool, dtype_out : np.dtype = None)->np.ndarray:
        """
        @public
        Resample image or segmentation data to a new shape, optionally resampling separately along an anisotropic axis

        Args:
            data (np.ndarray): Input array of shape (c, x, y, z)
            new_shape (np.ndarray): Future shape of the resempled data
            axis (int): The axis to resample separately, or None if not needed
            do_separate_z (bool): True if separate resampling along the z-axis should be done
            dtype_out (np.dtype, optional): Output data type. Defaults to None.

        Returns:
            np.ndarray: Output data array of shape (c, x, y, z)
        """
        assert data.ndim == 4, "data must be (c, x, y, z)"
        assert len(new_shape) == data.ndim - 1

        if self.is_seg:
            resize_fn = resize_segmentation
            kwargs = OrderedDict()
        else:
            resize_fn = resize
            kwargs = {'mode': 'edge', 'anti_aliasing': False}
        shape = np.array(data[0].shape)
        new_shape = np.array(new_shape)
        if dtype_out is None:
            dtype_out = data.dtype
        reshaped_final = np.zeros((data.shape[0], *new_shape), dtype=dtype_out)
        if np.any(shape != new_shape):
            data = data.astype(float, copy=False)
            if do_separate_z:
                assert axis is not None, 'If do_separate_z, we need to know what axis is anisotropic'
                if axis == 0:
                    new_shape_2d = new_shape[1:]
                elif axis == 1:
                    new_shape_2d = new_shape[[0, 2]]
                else:
                    new_shape_2d = new_shape[:-1]

                for c in range(data.shape[0]):
                    tmp = deepcopy(new_shape)
                    tmp[axis] = shape[axis]
                    reshaped_here = np.zeros(tmp)
                    for slice_id in range(shape[axis]):
                        if axis == 0:
                            reshaped_here[slice_id] = resize_fn(data[c, slice_id], new_shape_2d, self.order, **kwargs)
                        elif axis == 1:
                            reshaped_here[:, slice_id] = resize_fn(data[c, :, slice_id], new_shape_2d, self.order, **kwargs)
                        else:
                            reshaped_here[:, :, slice_id] = resize_fn(data[c, :, :, slice_id], new_shape_2d, self.order, **kwargs)
                    if shape[axis] != new_shape[axis]:

                        # The following few lines are blatantly copied and modified from sklearn's resize()
                        rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                        orig_rows, orig_cols, orig_dim = reshaped_here.shape

                        # align_corners=False
                        row_scale = float(orig_rows) / rows
                        col_scale = float(orig_cols) / cols
                        dim_scale = float(orig_dim) / dim

                        map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                        map_rows = row_scale * (map_rows + 0.5) - 0.5
                        map_cols = col_scale * (map_cols + 0.5) - 0.5
                        map_dims = dim_scale * (map_dims + 0.5) - 0.5

                        coord_map = np.array([map_rows, map_cols, map_dims])
                        if not self.is_seg or self.order_z == 0:
                            reshaped_final[c] = map_coordinates(reshaped_here, coord_map, order=self.order_z, mode='nearest')[None]
                        else:
                            unique_labels = np.sort(pd.unique(reshaped_here.ravel()))  # np.unique(reshaped_data)
                            for i, cl in enumerate(unique_labels):
                                reshaped_final[c][np.round(
                                    map_coordinates((reshaped_here == cl).astype(float), coord_map, order=self.order_z,
                                                    mode='nearest')) > 0.5] = cl
                    else:
                        reshaped_final[c] = reshaped_here
            else:
                for c in range(data.shape[0]):
                    reshaped_final[c] = resize_fn(data[c], new_shape, self.order, **kwargs)
            return reshaped_final
        else:
            return data
