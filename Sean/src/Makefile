CPP = g++
CXXFLAGS = -std=c++11 -O2 -Wall
TARGET = gen

ifeq ($(OS),Windows_NT)
	LIBS = -lopenblas
	LDFLAGS = -L../../openblas/lib
	INCLUDES = -I../../openblas/include
else
	LIBS= -lopenblas_threads
	LDFLAGS =
	INCLUDES =
endif

RM = rm

SRC = error.cpp pgm.cpp readraw.cpp preprocess.cpp main.cpp
OBJ = $(SRC:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CPP) $(CXXFLAGS) -o $(TARGET) $(OBJ) $(LDFLAGS) $(LIBS)

.cpp.o:
	$(CPP) $(CXXFLAGS) -c $< -o $@ $(INCLUDES)

.PHONY: clean
clean:
	$(RM) $(OBJ) $(TARGET)
