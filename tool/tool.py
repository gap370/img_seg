
import nibabel as nib
import numpy as np
import os

def seg_display(output_data, original_nifti, out_name='test_output'):
	print(original_nifti)
	orig_img = nib.load(original_nifti)
	orig_affine = orig_img.affine

	img_output = nib.Nifti1Image(output_data, orig_affine)
	img_output.to_filename(out_name + '_pred.nii.gz')

	print(out_name + 'saved.')

	
