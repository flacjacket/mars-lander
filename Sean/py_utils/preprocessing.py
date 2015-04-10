from __future__ import division, print_function

import numpy as np


def _smooth(df, smooth):
    df_out = np.zeros_like(df)
    for xi in range(df.shape[0]):
        for yi in range(df.shape[1]):
            df_out[xi, yi] = np.any(df[xi-smooth:xi+smooth, yi-smooth:yi+smooth])
    return df_out * 0xff


def check_angle(df_in, r=17, r_foot=2.5, angle=10, buffer=10, smooth=None):
    """Check the angle of the positions, returning a landing matrix for each
    point"""
    df_out = np.zeros_like(df_in, dtype='u1')

    angle = np.deg2rad(angle)

    # How to extract positions under the feet
    x = np.arange(-18, 19, 2)
    y = np.arange(0, 19, 2)
    X_foot, Y_foot = np.meshgrid(x, y)
    sel_foot = np.logical_and(
        X_foot**2 + Y_foot**2 >= (r - (2 * r_foot + 0.5))**2,
        X_foot**2 + Y_foot**2 <= (r + 0.5)**2
    )

    # Distance between foot positions
    spacing = 0.1
    x = -X_foot[sel_foot] * spacing
    y = -Y_foot[sel_foot] * spacing
    foot_dist = np.sqrt((2*x)**2 + (2*y)**2)

    for yi in range(buffer, df_in.shape[0]-buffer):
        for xi in range(buffer, df_in.shape[1]-buffer):
            Z = df_in[yi-(r+1)//2:yi+(r+3)//2, xi-(r+1)//2:xi+(r+3)//2]
            dZ_foot = (Z[(r+1)//2:, :] - Z[(r+1)//2::-1, ::-1])[sel_foot]

            if np.all(np.abs(np.arctan2(dZ_foot, foot_dist)) < angle):
                df_out[yi, xi] = 0xff

    if smooth:
        df_out = _smooth(df_out, smooth)

    df_out = np.hstack([df_out, df_out])
    df_out = np.vstack([df_out.flatten(), df_out.flatten()]).T.reshape(1000, 1000)
    df_out[20, :] = df_out[-20, :] = df_out[:, 20] = df_out[:, -20] = 0x00

    return df_out


def check_full(df_in, r=17, r_foot=2.5, angle=10, height=0.41, buffer=10, smooth=None):
    """Check the angle and collisions of the positions, returning a landing
    matrix for each point"""
    df_out = np.zeros_like(df_in, dtype='u1')

    angle = np.deg2rad(angle)

    # How to extract positions under the feet
    x = np.arange(-18, 19, 2)
    y = np.arange(0, 19, 2)
    X_foot, Y_foot = np.meshgrid(x, y)
    sel_foot = np.logical_and(
        X_foot**2 + Y_foot**2 >= (r - (2 * r_foot + 0.5))**2,
        X_foot**2 + Y_foot**2 <= (r + 0.5)**2
    )

    # Distance between foot positions
    spacing = 0.1
    x = -X_foot[sel_foot] * spacing
    y = -Y_foot[sel_foot] * spacing
    foot_dist = np.sqrt((2*x)**2 + (2*y)**2)

    # How to extract positions under the base
    x = np.arange(-18, 19, 2)
    y = np.arange(-18, 19, 2)
    X_base, Y_base = np.meshgrid(x, y)
    sel_base = X_base**2 + Y_base**2 <= (r + 0.5)**2

    for xi in range(buffer, df_in.shape[0]-buffer):
        for yi in range(buffer, df_in.shape[1]-buffer):
            Z = df_in[yi-(r+1)//2:yi+(r+3)//2, xi-(r+1)//2:xi+(r+3)//2]
            dZ_foot = (Z[(r+1)//2::-1, ::-1] - Z[(r+1)//2:, :])[sel_foot]
            _angle = np.abs(np.arctan2(dZ_foot, foot_dist))

            if np.any(_angle > angle):
                continue

            ind = _angle.argmin()

            x = X_foot[sel_foot][ind]
            y = Y_foot[sel_foot][ind]

            ax = -2 * x * spacing
            ay = -2 * y * spacing
            z = Z[(r+1)//2:][sel_foot][ind]

            a = np.array([[ax, ay, dZ_foot[ind]]])

            b = np.vstack([(X_base[sel_base] - x) * spacing,
                           (Y_base[sel_base] - y) * spacing,
                           Z[sel_base] - z]).T

            phi = np.arctan2(ay, ax)
            rot = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])

            a = np.dot(a, rot)
            b = np.dot(b, rot)

            if np.max(a[:, 0] * b[:, 2] - a[:, 2] * b[:, 0]) / np.linalg.norm(a) < height:
                df_out[xi, yi] = 0xff

    if smooth:
        df_out = _smooth(df_out, smooth)

    df_out = np.hstack([df_out, df_out])
    df_out = np.vstack([df_out.flatten(), df_out.flatten()]).T.reshape(1000, 1000).T

    return df_out
