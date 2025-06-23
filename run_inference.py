import torch
from inference.unet import ResidualEncoderUNet
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

network = ResidualEncoderUNet(
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

def compute_steps(image_size, patch_size, step_size):
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

def compute_gaussian(patch_size, sigma_scale, value_scaling_factor, dtype, device):
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)

    gaussian_importance_map = torch.from_numpy(gaussian_importance_map)

    gaussian_importance_map /= (torch.max(gaussian_importance_map) / value_scaling_factor)
    gaussian_importance_map = gaussian_importance_map.to(device=device, dtype=dtype)
    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    mask = gaussian_importance_map == 0
    gaussian_importance_map[mask] = torch.min(gaussian_importance_map[~mask])
    return gaussian_importance_map

network.initialize(network)
checkpoint = torch.load("./model.pth", map_location="cpu", weights_only=False)
network.load_state_dict(checkpoint["network_weights"])
print("Checkpoint loaded successfully.")
network.to(device)
network.eval()
img = nib.load("./sub-r001s001_0000.nii.gz")
affine = img.affine
img=img.get_fdata().astype("float32")
patch_size=[128, 128, 128]
steps = compute_steps(img.shape, patch_size, 0.5)
total_patches = len(steps[0]) * len(steps[1]) * len(steps[2])
count = 0
print(steps)
output_sum = torch.zeros(1,2,*img.shape)
for x in steps[0]:
    for y in steps[1]:
        for z in steps[2]:
            count+=1
            patch = img[x:x+patch_size[0],y:y+patch_size[1],z:z+patch_size[2]]
            patch=torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = network(patch)
                output_sum[:, :, x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]] += pred.cpu()
            print(f"Fin traitement du patch {count} / {total_patches}")

output = output_sum.numpy()
out_img = nib.Nifti1Image(output,affine)
nib.save(out_img, "output.nii.gz")