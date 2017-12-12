from pyquaternion import Quaternion
import math

# pitch: y-axis
# rool : x-axis
# yaw  : z-axis
#
#
def quaternion_from_euler(pitch, roll, yaw):
    cy = math.cos(yaw * 0.5);
    sy = math.sin(yaw * 0.5);
    cr = math.cos(roll * 0.5);
    sr = math.sin(roll * 0.5);
    cp = math.cos(pitch * 0.5);
    sp = math.sin(pitch * 0.5);

    return Quaternion(cy * cr * cp + sy * sr * sp,
                      cy * sr * cp - sy * cr * sp,
                      cy * cr * sp + sy * sr * cp,
                      sy * cr * cp - cy * sr * sp)