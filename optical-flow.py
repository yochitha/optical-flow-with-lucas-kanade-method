import numpy as np
import cv2

def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. Here we assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use
                      'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1 because the autograder does not
                       use this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """

    ix = cv2.Sobel(img_a, cv2.CV_64F, 1, 0, ksize=3, scale=1/8)
    iy = cv2.Sobel(img_a, cv2.CV_64F, 0, 1, ksize=3, scale=1/8)

    ixix = ix * ix
    iyiy = iy * iy
    ixiy = ix * iy
    it = img_b - img_a
    ixit = ix * it
    iyit = iy * it

    ixix_g = cv2.GaussianBlur(ixix, (k_size, k_size), 0)
    iyiy_g = cv2.GaussianBlur(iyiy, (k_size, k_size), 0)
    ixiy_g = cv2.GaussianBlur(ixiy, (k_size, k_size), 0)
    ixit_g = cv2.GaussianBlur(ixit, (k_size, k_size), 0)
    iyit_g = cv2.GaussianBlur(iyit, (k_size, k_size), 0)

    a = np.array([[ixix_g, ixiy_g], [ixiy_g, iyiy_g]])
    b = np.array([- ixit_g, - iyit_g])

    A = np.moveaxis(a, [0, 1, 2, 3], [2, 3, 0, 1])
    B = np.moveaxis(b, [0, 1, 2], [2, 0, 1])
    shape = img_b.shape
    u = np.zeros(shape)
    v = np.zeros(shape)

    det = A[np.nonzero(np.linalg.det(A) > 0.0)]
    beta = B[np.nonzero(np.linalg.det(A) > 0.0)]

    res = np.linalg.solve(det, beta)

    dims = np.transpose(np.where(np.linalg.det(A) > 0.0))

    for (pixel, value) in zip(dims, res):
        u[pixel[0], pixel[1]] = value[0]
        v[pixel[0], pixel[1]] = value[1]

    return u, v
  
  
def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation,
                    border_mode):
    """Computes the optic flow using Hierarchical Lucas-Kanade.

    This method should use reduce_image(), expand_image(), warp(),
    and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis,
                             same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along Y-axis,
                             same size and type as U.
    """
    pyr1 = gaussian_pyramid(img_a, levels)
    pyr2 = gaussian_pyramid(img_b, levels)

    img1 = pyr1[-1]
    img2 = pyr2[-1]

    u, v = optic_flow_lk(img1, img2, k_size, k_type, sigma)

    for i in range(levels - 2, -1, -1):
        u_exp = expand_image(u) * 2
        v_exp = expand_image(v) * 2
        warp_img = warp(pyr2[i], u_exp, v_exp, interpolation, border_mode)
        u1, v1 = optic_flow_lk(warp_img, pyr1[i], k_size, k_type, sigma)
        delta_u, delta_v = u_exp - u1, v_exp - v1
        u, v = delta_u, delta_v

    return u, v
