from skeleton import NodeType as NT
import glm
import collections
import functools


PartRenderData = collections.namedtuple("PartRenderData",
                                        "scale_func enabled color")
def get_squared_character_render_data():
    def non_uniform_scale_func(sx, sy, sz, length):
        return glm.vec3(length * sx,
                        length * sy,
                        length * sz)

    default_scale = 0.8
    def y_axis_scale_func(fixed_scale, length):
        return glm.vec3(fixed_scale,
                        max(fixed_scale, length),
                        fixed_scale)

    RENDER_DATA = {}
    RENDER_DATA[NT.HEAD] = PartRenderData(
        scale_func = functools.partial(y_axis_scale_func, default_scale),
        enabled = True,
        color = glm.vec4(213/255, 0.0, 249.0/255, 1.0)
    )
    RENDER_DATA[NT.EYE] = PartRenderData(
        scale_func = functools.partial(y_axis_scale_func, default_scale),
        enabled = False,
        color = glm.vec4(1.0, 1.0, 1.0, 1.0)
    )
    RENDER_DATA[NT.NECK] = PartRenderData(
        scale_func = functools.partial(y_axis_scale_func, default_scale),
        enabled = True,
        color = glm.vec4(213/255, 0.0, 249.0/255, 1.0)
    )
    RENDER_DATA[NT.SPINE] = PartRenderData(
        scale_func = functools.partial(y_axis_scale_func, default_scale),
        enabled = True,
        color = glm.vec4(1.0, 0.0 / 255, 7.0 / 255, 1.0)
    )
    RENDER_DATA[NT.TORSO] = PartRenderData(
        scale_func = functools.partial(y_axis_scale_func, default_scale),
        enabled = True,
        color = glm.vec4(1.0, 193.0 / 255, 7.0 / 255, 1.0)
    )
    RENDER_DATA[NT.UPPER_LEG] = PartRenderData(
        scale_func = functools.partial(y_axis_scale_func, default_scale),
        enabled = True,
        color = glm.vec4(63.0/255, 81.0/255, 181.0/255, 1.0)
    )
    RENDER_DATA[NT.LOWER_LEG] = PartRenderData(
        scale_func = functools.partial(y_axis_scale_func, default_scale),
        enabled = True,
        color = glm.vec4(63.0/255, 81.0/255, 181.0/255, 1.0)
    )
    RENDER_DATA[NT.FOOT] = PartRenderData(
        scale_func = functools.partial(y_axis_scale_func, default_scale),
        enabled = True,
        color = glm.vec4(26.0/255, 35.0/255, 126.0/255, 1.0)
    )
    RENDER_DATA[NT.UPPER_ARM] = PartRenderData(
        scale_func = functools.partial(y_axis_scale_func, default_scale),
        enabled = True,
        color = glm.vec4(63.0/255, 81.0/255, 181.0/255, 1.0)
    )
    RENDER_DATA[NT.LOWER_ARM] = PartRenderData(
        scale_func = functools.partial(y_axis_scale_func, default_scale),
        enabled = True,
        color = glm.vec4(63.0/255, 81.0/255, 181.0/255, 1.0)
    )
    RENDER_DATA[NT.FINGER] = PartRenderData(
        scale_func = functools.partial(y_axis_scale_func, default_scale),
        enabled = True,
        color = glm.vec4(26.0/255, 35.0/255, 126.0/255, 1.0)
    )
    RENDER_DATA[NT.HAND] = PartRenderData(
        scale_func = functools.partial(y_axis_scale_func, default_scale),
        enabled = True,
        color = glm.vec4(26.0/255, 35.0/255, 126.0/255, 1.0)
    )
    return RENDER_DATA
