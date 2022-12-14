import cv2
import os
import numpy as np
import optical-flow

input_dir = "input_images"
output_dir = "./"

def quiver(u, v, scale, stride, color=(0, 255, 0)):
    img_out = np.zeros((v.shape[0], u.shape[1], 3), dtype=np.uint8)
    for y in range(0, v.shape[0], stride):
        for x in range(0, u.shape[1], stride):
            cv2.line(img_out, (x, y), (x + int(u[y, x] * scale),
                                       y + int(v[y, x] * scale)), color, 1)
            cv2.circle(img_out, (x + int(u[y, x] * scale),
                                 y + int(v[y, x] * scale)), 1, color, 1)
    return img_out
  
def run_method():
    """ You will compare the base image Shift0.png with the remaining
    images located in the directory TestSeq:
    - ShiftR10.png
    - ShiftR20.png
    - ShiftR40.png

    Returns:
        None
    """
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR10.png'), 0) / 255.
    shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR20.png'), 0) / 255.
    shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR40.png'), 0) / 255.

    shift_0 = cv2.GaussianBlur(shift_0, (31, 31), 0)
    shift_1_0 = cv2.GaussianBlur(shift_0, (75, 75), 0)
    shift_2_0 = cv2.GaussianBlur(shift_0, (101, 101), 0)
    shift_r10 = cv2.GaussianBlur(shift_r10, (31, 31), 0)
    shift_r20 = cv2.GaussianBlur(shift_r20, (75, 75), 0)
    shift_r40 = cv2.GaussianBlur(shift_r40, (101, 101), 0)

    k_size = 151
    k_type = "gaussian"
    sigma = 0
    u, v = optic-flow.optic_flow_lk(shift_0, shift_r10, k_size, k_type, sigma)
    u_v = quiver(u, v, scale=1.2, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-1.png"), u_v)

    k_size = 151
    k_type = "gaussian"
    sigma = 0
    u, v = optic-flow.optic_flow_lk(shift_1_0, shift_r20, k_size, k_type, sigma)
    u_v = quiver(u, v, scale=0.4, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-2.png"), u_v)

    k_size = 151
    k_type = "gaussian"
    sigma = 0
    u, v = optic-flow.optic_flow_lk(shift_2_0, shift_r40, k_size, k_type, sigma)
    u_v = quiver(u, v, scale=0.2, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-3.png"), u_v)

    return None
  
if __name__ == '__main__':
  run_method()
