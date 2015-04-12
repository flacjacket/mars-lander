CXXFLAGS = -std=c++11 -O2 -Wall -Wextra -pedantic
GEN_TARGETS = gen_easy gen_full
ifeq ($(OS),Windows_NT)
	CPP = g++
	LIBS = -lopenblas
	LDFLAGS = -L../../openblas/lib
	INCLUDES = -I../../openblas/include
	EXECUTABLES = $(GEN_TARGETS:%=%.exe)
else
	CPP = clang++
	LIBS= -lopenblas_threads
	LDFLAGS =
	INCLUDES =
	EXECUTABLES = $(GEN_TARGETS)
endif

SRC = error.cpp pgm.cpp preprocess_common.cpp raw.cpp
OBJ = $(SRC:.cpp=.o)

all: $(GEN_TARGETS)

gen_%: gen_%.o nn_gen.o preprocess_%.o $(OBJ)
	$(CPP) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LIBS)

gen_%.o: gen_%.cpp
	$(CPP) $(CXXFLAGS) -c $< -o $@ $(INCLUDES)

%.o: %.cpp %.h
	$(CPP) $(CXXFLAGS) -c $< -o $@ $(INCLUDES)

.SECONDARY: $(OBJS)

.PHONY: clean
clean:
	-rm $(OBJ) $(GEN_TARGETS:%=%.o) $(GEN_TARGETS:gen_%=preprocess_%.o) $(EXECUTABLES)
