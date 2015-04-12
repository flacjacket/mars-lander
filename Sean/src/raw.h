/*********************************************************************
 * readraw.h
 ********************************************************************/

#ifndef _READRAW_H
#define _READRAW_H

namespace raw {
    std::vector<float> read_file(const char *filename, std::size_t size);
}

#endif // _READRAW_H
