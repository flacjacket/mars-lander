CXXFLAGS = -std=c++11 -O2 -Wall -Wextra -pedantic
ifeq ($(OS),Windows_NT)
	CPP = g++
	LIBS = -lopenblas
	LDFLAGS = -L../../openblas/lib
	INCLUDES = -I../../openblas/include
else
	CPP = clang++
	LIBS= -lopenblas_threads
	LDFLAGS =
	INCLUDES =
endif

RM = rm

SRC = error.cpp pgm.cpp readraw.cpp preprocess.cpp
OBJ = $(SRC:.cpp=.o)

all: gen_easy

gen_simple: gen_simple.o $(OBJ)
	$(CPP) $(CXXFLAGS) -o gen_simple gen_simple.o $(OBJ) $(LDFLAGS) $(LIBS)

gen_easy: gen_easy.o $(OBJ)
	$(CPP) $(CXXFLAGS) -o gen_easy gen_easy.o $(OBJ) $(LDFLAGS) $(LIBS)

.cpp.o:
	$(CPP) $(CXXFLAGS) -c $< -o $@ $(INCLUDES)

.PHONY: clean
clean:
	$(RM) $(OBJ) *.exe
