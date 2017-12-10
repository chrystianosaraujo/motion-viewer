from pyquaternion import Quaternion
import math

def quaternion_from_euler(alpha, beta, gamma):
    ca = math.cos(alpha * 0.5)
    cb = math.cos(beta  * 0.5)
    cc = math.cos(gamma * 0.5)
    sa = math.sin(alpha * 0.5)
    sb = math.sin(beta  * 0.5)
    sc = math.sin(gamma * 0.5)

    return Quaternion(ca * cb * cc - sa * cb * sc,
                      ca * sb * sc - sa * sb * cc,
                      ca * sb * cc + sa * sb * sc,
                      sa * cb * cc + ca * cb * sc)
