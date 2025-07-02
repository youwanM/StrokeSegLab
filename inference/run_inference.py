import logging
from inference.unet import ResidualEncoderUNet
from manager.option_manager import Option
import torch
import numpy as np
from scipy.ndimage import gaussian_filter
import time

class Inference:
    def __init__(self,gui=None,patch_size=[128,128,128]):
        self.logger=logging.getLogger()
        option = Option()
        self.device = option.get("device","cpu")
        self.gui = gui

        self.network = ResidualEncoderUNet(
            input_channels=1,
            n_stages=6,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=torch.nn.Conv3d,
            kernel_sizes=[[3, 3, 3] for _ in range(6)],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
            num_classes=2,
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
            conv_bias=True,
            norm_op=torch.nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            nonlin=torch.nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=False,
        )
        self.network.initialize()
        model_path = option.get("model_path","./models/model.pth")
        self.logger.debug(f'model path : {model_path}')
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.network.load_state_dict(checkpoint["network_weights"])
        self.logger.info("Checkpoint loaded successfully.")
        self.network.to(self.device)
        self.network.eval()
        self.patch_size = patch_size
    
    def _compute_steps(self,image_size, patch_size, step_size =0.5):
        assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
        assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

        # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
        # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
        target_step_sizes_in_voxels = [i * step_size for i in patch_size]

        num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]

        steps = []
        for dim in range(len(patch_size)):
            # the highest step value for this dimension is
            max_step_value = image_size[dim] - patch_size[dim]
            if num_steps[dim] > 1:
                actual_step_size = max_step_value / (num_steps[dim] - 1)
            else:
                actual_step_size = 99999999999  # does not matter because there is only one step at 0

            steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

            steps.append(steps_here)

        return steps
    
    def _compute_gaussian(self,patch_size, dtype=torch.float16,sigma_scale=1./8, value_scaling_factor=10):
        tmp = np.zeros(patch_size)
        center_coords = [i // 2 for i in patch_size]
        sigmas = [i * sigma_scale for i in patch_size]
        tmp[tuple(center_coords)] = 1
        gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)

        gaussian_importance_map = torch.from_numpy(gaussian_importance_map)

        gaussian_importance_map /= (torch.max(gaussian_importance_map) / value_scaling_factor)
        gaussian_importance_map = gaussian_importance_map.to(device=self.device, dtype=dtype)
        # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
        mask = gaussian_importance_map == 0
        gaussian_importance_map[mask] = torch.min(gaussian_importance_map[~mask])
        return gaussian_importance_map

    def run(self,data):
        assert data.ndim == 4, f"Expected 4D input (c, x, y, z), got {data.shape}"
        _, x, y, z = data.shape
        steps = self._compute_steps((x, y, z),self.patch_size)
        gaussian = self._compute_gaussian(self.patch_size)
        total_patches = len(steps[0]) * len(steps[1]) * len(steps[2])
        count = 0
        output = torch.zeros(1,2, x, y, z)
        normalization_map = torch.zeros(1, 1, x, y, z)
        start_time = time.time()
        last_time = start_time
        for x in steps[0]:
            for y in steps[1]:
                for z in steps[2]:
                    count+=1
                    patch = data[:,x:x+self.patch_size[0],y:y+self.patch_size[1],z:z+self.patch_size[2]]
                    patch=torch.from_numpy(patch).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        pred = self.network(patch)*gaussian
                        output[:, :, x:x+self.patch_size[0], y:y+self.patch_size[1], z:z+self.patch_size[2]] += pred.cpu()
                        normalization_map[:, :, x:x+self.patch_size[0], y:y+self.patch_size[1], z:z+self.patch_size[2]] += gaussian.cpu()
                    now = time.time()
                    total_elapsed = now - start_time
                    iter_time = now - last_time
                    remaining = (total_elapsed / count) * (total_patches - count)
                    self.logger.info(f"Patch {count}/{total_patches} done | " f"Iter time: {iter_time:.2f}s | " f"Total: {total_elapsed:.2f}s | " f"ETA: {remaining:.2f}s")
                    if(self.gui !=None):
                        self.gui.update_status(f"Patch {count}/{total_patches} done | " f"Iter time: {iter_time:.2f}s | " f"Total: {total_elapsed:.2f}s | " f"ETA: {remaining:.2f}s")
                    last_time = now
        output = output.numpy()
        output /= normalization_map.numpy()
        return output
