#include <algorithm>

#include "readraw.h"

#define ZH 10
#define ZW 19

int main() {
    std::array<float, NROWS*NCOLS> data;
    std::array<float, ZH*ZW> z1;
    std::array<float, ZH*ZW> z2;

    data = read_raw("raw.dem");
    for (int i = 10; i < NROWS - 10; i++) {
        for (int j = 10; j < NCOLS - 10; j++) {
            for (int k = 0; k < ZH; k++) {
                std::copy(&data[(i-(ZH-1)+k)*NCOLS + j - ZH+1],
                          &data[(i-(ZH-1)+k)*NCOLS + j + ZH],
                          &z1[k * ZW]);
                std::reverse_copy(&data[(i+(ZH-1)-k)*NCOLS + j - ZH+1],
                                  &data[(i+(ZH-1)-k)*NCOLS + j + ZH],
                                  &z2[k * ZW]);
            }
        }
    }
}
