from __future__ import print_function, division

import numpy as np
import scipy.misc

SAFE = 0xff
UNSAFE = 0x00


def generate_diff(df1, df2):
    """Compares the first array to the second array"""
    assert df1.dtype == df2.dtype, "Input arrays must have matching type"
    assert df1.shape == df2.shape, "Input arrays must have matching shape"

    s = list(df1.shape) + [3]

    df_out = np.empty(s)

    # both match
    cor_safe = np.logical_and(df1 == SAFE, df2 == SAFE)
    cor_unsafe = np.logical_and(df1 == UNSAFE, df2 == UNSAFE)
    good = np.logical_or(cor_safe, cor_unsafe)
    # categorized safe, but is unsafe
    inc_safe = np.logical_and(df1 == SAFE, df2 == UNSAFE)
    # categorized unsafe, but is safe
    inc_unsafe = np.logical_and(df1 == UNSAFE, df2 == SAFE)

    # copy good data over
    df_out[good, :] = (np.repeat(df1, 3).reshape(s))[good, :]

    # mark improperly safe as red
    df_out[inc_safe, 0] = 0xff
    # mark improperly unsafe as blue
    df_out[inc_unsafe, 2] = 0xff

    cor_safe = np.sum(cor_safe)
    cor_unsafe = np.sum(cor_unsafe)
    inc_safe = np.sum(inc_safe)
    inc_unsafe = np.sum(inc_unsafe)

    return df_out, (cor_safe, cor_unsafe, inc_safe, inc_unsafe)


def compare_output(df_gen, df_cmp, output):
    df_diff, results = generate_diff(df_gen, df_cmp)

    scipy.misc.imsave(output, df_diff)

    def f_score(tp, tn, fp, fn):
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        return 2 * prec * rec / (prec + rec)

    f1 = f_score(results[0], results[1], results[2], results[3])
    f2 = f_score(results[1], results[0], results[3], results[2])

    total = sum(results)
    total_safe = results[0] + results[3]  # correct safe + incorrect unsafe
    total_unsafe = results[1] + results[2]  # correct unsafe + incorrect safe

    per_correct = 100 * (results[0] + results[1]) / total
    # per_inc_safe = 100 * results[2] / total
    # per_inc_unsafe = 100 * results[3] / total

    per_safe = 100 * results[0] / total_safe
    per_unsafe = 100 * results[1] / total_unsafe

    print("Accuracy:          %4.1f%%" % per_correct)
    print("F1 (safe):         %4.2f" % f1)
    print("F1 (unsafe):       %4.2f" % f2)
    # print("Incorrectly classified safe:   %4.1f%%" % per_inc_safe)
    # print("Incorrectly classified unsafe: %4.1f%%" % per_inc_unsafe)
    print()
    print("Safe classification accuracy:   %4.1f%% (%d / %d)" %
          (per_safe, results[0], total_safe))
    print("Unsafe classification accuracy: %4.1f%% (%d / %d)" %
          (per_unsafe, results[1], total_unsafe))
