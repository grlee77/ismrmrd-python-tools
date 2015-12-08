import numpy as np
from ismrmrdtools import transform


def cart_pattern_3d(img_shape, acc=(1, 1), ref=(0, 0), shift=(0, 0),
                    caipi_yshift=0):
    """Generate a pattern undersampled in two dimensions.  For use with 3D
    Cartesian acquisitions where two phase encode axes exist.

    Paramters
    ---------
    img_shape : 2-tuple or (y, x) array
        Size of the sampling pattern to generate.
    acc : 2-tuple, optional
        Acceleration factors.  Currently must be integers.
    ref : 2-tuple, optional
        Size of fully sampled reference region (in center of k-space).
    shift : 2-tuple, optional
        Offset the sampling pattern by this amount on each axis (periodic).
    caipi_yshift : int, optional
        y-shift at position x will be: ``(shift[0] + caipi_yshift*x) % acc[0]``

    Returns
    -------
    data : (c, ky, kx) array
        Sampled data in k-space (zeros where not sampled).
    pat : (c, ky, kx) array
        Sampling pattern : (0 = not sampled,
                            1 = imaging data,
                            2 = reference data,
                            3 = reference and imaging data)

    """

    if not np.all([len(a) == 2 for a in [img_shape, acc, ref, shift]]):
        raise ValueError("expected inputs: img_shape, acc, ref, and shift to "
                         "all be length 2")
    ny, nx = img_shape
    acc_y, acc_x = acc
    ref_y, ref_x = ref
    shift_x = shift[0] % acc_y
    shift_y = shift[1] % acc_x

    if np.any(np.mod(acc, 1) > 0):
        raise ValueError("only integer acceleration factors supported")
    if np.any(np.mod(ref, 1) > 0):
        raise ValueError("reference region dimensions must be integers")
    if np.any(np.mod(shift, 1) > 0):
        raise ValueError("sample pattern shifts must be integers")
    if (caipi_yshift % 1) > 0:
        raise ValueError("CAIPIRINHA shift must be an integer")

    pat_img = np.zeros(img_shape, dtype=np.int8)
    if caipi_yshift == 0:
        pat_img[shift_y:ny:acc_y, shift_x:nx:acc_x] = 1
    else:
        for x in range(shift_x, nx, acc_x):
            caipi_shift = (shift_y + caipi_yshift*x//acc_x) % acc_y
            pat_img[slice(caipi_shift, ny, acc_y), x] = 1

    if ref_y > 0 or ref_x > 0:
        pat_ref = np.zeros(img_shape, dtype=np.int8)
        if ref_y > 0 and ref_x > 0:
            pat_ref[(0+ny//2):(ref_y+ny//2),
                    (0+nx//2):(ref_x+nx//2)] = 2
        elif ref_y > 0:
            pat_ref[(0+ny//2):(ref_y+ny//2), :] = 2
        elif ref_x > 0:
            pat_ref[:, (0+nx//2):(ref_x+nx//2)] = 2
        pat = pat_img + pat_ref
    else:
        pat = pat_img
    return pat


def factors(n):
    """ Return all integer factors of n. """
    if n % 1 != 0:
        raise ValueError("n must be an integer")
    f = n / np.arange(1, np.ceil(np.sqrt(n))+1)
    f = f[f == np.floor(f)].astype(np.int)
    return np.unique(np.concatenate((f, n//f), axis=0))


def show_all_caipi_patterns(R=6):
    """ Plot all CAIPIRINHA patterns corresponding to a given acceleration
    factor. """
    from matplotlib import pyplot as plt
    shape = (R, R)
    Rvals_y = factors(R)  # all possible Ry values

    # initialize grid of subplots
    gridspec_kw = dict(hspace=0.05, wspace=0.05*R/len(Rvals_y))
    fig, axes = plt.subplots(len(Rvals_y), R, gridspec_kw=gridspec_kw)
    fig.set_size_inches(np.array([8*R/len(Rvals_y), 8]), forward=True)

    for iy, acc_y in enumerate(Rvals_y):
        acc_x = R // acc_y
        # valid CAIPI shifts are in the range [0, acc_y-1]
        for ix, caipi_yshift in enumerate(np.arange(acc_y)):
            pat = cart_pattern_3d(shape, acc=(acc_y, acc_x),
                                  caipi_yshift=caipi_yshift)

            # show pattern at the appropriate subplot
            axes[iy, ix].imshow(pat[:R, :R], interpolation='nearest',
                                cmap=plt.cm.gray)
            if ix == 0:
                axes[iy, ix].set_ylabel(r'$R_y$={}, $R_x$={}'.format(acc_y,
                                                                     acc_x))
            if iy == len(Rvals_y)-1:
                axes[iy, ix].set_xlabel(r'$\Delta$={}'.format(caipi_yshift))
            axes[iy, ix].set_xticklabels([])
            axes[iy, ix].set_yticklabels([])
        for ix in np.arange(acc_y, R):
            axes[iy, ix].set_axis_off()
            axes[iy, ix].set_xticklabels([])
            axes[iy, ix].set_yticklabels([])


def sample_data_3d(img_obj, csm, acc=(1, 1), ref=(0, 0), shift=(0, 0),
                   caipi_yshift=0):
    """Sample the k-space of object provided in `img_obj` after first applying
    coil sensitivity maps in `csm` and Fourier transforming to k-space.

    Paramters
    ---------
    img_obj : (y, x) array
        Object in image space
    csm : (c, y, x) array
        Coil sensitivity maps
    acc : float, optional
        Acceleration factor
    ref : float, optional
        Reference lines (in center of k-space)
    sshift : float, optional
        Sampling shift, i.e for undersampling, do we start with line 1 or line
        1+sshift?.

    Returns
    -------
    data : (c, ky, kx) array
        Sampled data in k-space (zeros where not sampled).
    pat : (c, ky, kx) array
        Sampling pattern : (0 = not sampled,
                            1 = imaging data,
                            2 = reference data,
                            3 = reference and imaging data)

    Notes
    -----
    Code made available for the ISMRM 2013 Sunrise Educational Course

    Michael S. Hansen (michael.hansen@nih.gov)
    """

    if img_obj.ndim != 2:
        raise ValueError("Only two dimensional objects supported at the "
                         "moment")
    if csm.ndim != 3:
        raise ValueError("csm must be a 3 dimensional array")
    if img_obj.shape[0:2] != csm.shape[1:3]:
        raise ValueError("Object and csm dimension mismatch")

    pat = cart_pattern_3d(img_obj.shape, acc, ref, shift, caipi_yshift)

    coil_images = img_obj[np.newaxis, ...] * csm
    data = transform.transform_image_to_kspace(coil_images, dim=(1, 2))
    data = data * (np.tile(pat, (csm.shape[0], 1, 1)) > 0).astype('float32')
    return (data, pat)
