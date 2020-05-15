import SimpleITK as sitk


def write_itk(arr, origin, spacing, out_path):
    itk_img = sitk.GetImageFromArray(arr)
    itk_img.SetOrigin(origin)
    itk_img.SetSpacing(spacing)
    sitk.WriteImage(itk_img, out_path)
