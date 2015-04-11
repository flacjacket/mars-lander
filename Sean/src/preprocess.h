/*********************************************************************
 * preprocess.h
 ********************************************************************/

#ifndef _PREPROCESS_H
#define _PREPROCESS_H

std::vector<unsigned char> preprocess_easy(std::vector<float> &data);
std::vector<unsigned char> preprocess_full(std::vector<float> &data);

void preprocess_fix_edges(std::vector<unsigned char> &output);

#endif // _PREPROCESS_H
