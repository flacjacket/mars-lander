/*********************************************************************
 * preprocess.h
 ********************************************************************/

#ifndef _PREPROCESS_H
#define _PREPROCESS_H

std::vector<unsigned char> preprocess_angle(std::vector<float> &data);

void fix_edges(std::vector<unsigned char> &output);

#endif // _PREPROCESS_H
