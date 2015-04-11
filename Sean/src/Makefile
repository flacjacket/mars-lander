CXXFLAGS = -std=c++11 -O2 -Wall -Wextra -pedantic
TARGETS = gen_simple gen_full
ifeq ($(OS),Windows_NT)
	CPP = g++
	LIBS = -lopenblas
	LDFLAGS = -L../../openblas/lib
	INCLUDES = -I../../openblas/include
	EXECUTABLES = $(TARGETS:%=%.exe)
else
	CPP = clang++
	LIBS= -lopenblas_threads
	LDFLAGS =
	INCLUDES =
	EXECUTABLES = $(TARGETS)
endif

SRC = error.cpp pgm.cpp readraw.cpp preprocess.cpp
OBJ = $(SRC:.cpp=.o)

all: $(TARGETS)

gen_%: gen_%.o $(OBJ)
	$(CPP) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)

gen_%.o: gen_%.cpp
	$(CPP) $(CXXFLAGS) -c $< -o $@ $(INCLUDES)

%.o: %.cpp %.h
	$(CPP) $(CXXFLAGS) -c $< -o $@ $(INCLUDES)

.SECONDARY: $(OBJS)

.PHONY: clean
clean:
	-rm $(OBJ) $(EXECUTABLES)
