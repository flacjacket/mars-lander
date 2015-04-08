import numpy as np


def check_angle(df_in, r=17, r_foot = 2.5, angle=10, buffer=10, smooth=None):
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

    for xi in range(buffer, df_in.shape[0]-buffer):
        for yi in range(buffer, df_in.shape[1]-buffer):
            df = df_in[xi-(r+1)//2:xi+(r+3)//2, yi-(r+1)//2:yi+(r+3)//2]
            Z = (df[(r+1)//2::-1, :] - df[(r+1)//2:, ::-1])[sel_foot]

            if np.all(np.abs(np.arctan2(Z, foot_dist)) < angle):
                df_out[xi, yi] = 0xff

    if smooth:
        df_out_new = np.zeros_like(df_out)
        for xi in range(df_in.shape[0]):
            for yi in range(df_in.shape[1]):
                df_out_new[xi, yi] = np.any(df_out[xi-smooth:xi+smooth, yi-smooth:yi+smooth])
        df_out = df_out_new * 0xff

    df_out = np.hstack([df_out, df_out])
    df_out = np.vstack([df_out.flatten(), df_out.flatten()]).T.reshape(1000, 1000)

    return df_out
