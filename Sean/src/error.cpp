/*********************************************************************
 * error.cpp
 *
 * Error and warning messages, and system commands.
 *********************************************************************/

#include <cstdarg>  // variadic function calls
#include <cstdlib>  // exit
#include <cstdio>   // vsprintf
#include <iostream>

#define MAXBUF 1024
/*********************************************************************
 * error
 *
 * Prints an error message and dies.
 *
 * INPUTS
 * exactly like printf
 */
void error(const char *fmt, ...) {
    va_list args;
    char buf[MAXBUF];

    va_start(args, fmt);
    vsprintf(buf, fmt, args);
    va_end(args);

    std::cerr << "Error: " << buf << std::endl;
    exit(1);
}


/*********************************************************************
 * warning
 *
 * Prints a warning message.
 *
 * INPUTS
 * exactly like printf
 */

void warning(const char *fmt, ...) {
    va_list args;
    char buf[MAXBUF];

    va_start(args, fmt);
    vsprintf(buf, fmt, args);
    va_end(args);

    std::cerr << "Warning: " << buf << std::endl;
}

