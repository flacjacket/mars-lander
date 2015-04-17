/*********************************************************************
 * preprocess.h
 ********************************************************************/

#ifndef _PREPROCESS_COMMON_H
#define _PREPROCESS_COMMON_H

#define SAFE 0xff
#define UNSAFE 0x00
#define FEED_TO_NET (0xff / 2)

// Use 10 deg as high safe cutoff
#define ANGLE_SAFE (10.*3.1415/180.)
// Use 13 deg as low unsafe cutoff
#define ANGLE_UNSAFE (13.*3.1415/180.)

// Use 0.29 m as high safe cutoff
#define HEIGHT_SAFE (HEIGHT_UNSAFE - 0.1)
// Use 0.39 m as low unsafe cutoff
#define HEIGHT_UNSAFE 0.39

#define ZH 11
#define ZW 21

void base_loc(double r_min, double r_max, std::vector<int> &d_loc);
void footpad_dist_4point(double r_min, double r_max, std::vector<float> &dist, std::vector<unsigned> &d_loc);

std::vector<unsigned char> preprocess_gen_pgm(std::vector<unsigned char> &output);

#endif // _PREPROCESS_COMMON_H
