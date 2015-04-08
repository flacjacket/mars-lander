CPP = g++
CXXFLAGS = -std=c++11 -O3 -Wall
TARGET = gen
LIBS= -lopenblas_threads
RM = rm

SRC = error.cpp pgm.cpp readraw.cpp preprocess.cpp main.cpp
OBJ = $(SRC:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CPP) $(CXXFLAGS) -o $(TARGET) $(OBJ) $(LIBS)

.cpp.o:
	$(CPP) $(CXXFLAGS) -c $< -o $@

.PHONY: clean
clean:
	$(RM) $(OBJ) $(TARGET)
