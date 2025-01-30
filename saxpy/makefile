# Compiler settings
CC = gcc
CFLAGS = -O3 -Wall -std=c99 -pthread -lm

# Target executable name
TARGET = saxpy

# Default target
all: $(TARGET)

# Compile source to executable
$(TARGET): saxpy.c
	$(CC) $(CFLAGS) -o $(TARGET) $^

# Clean up build files
clean:
	rm -f $(TARGET)

# Phony targets
.PHONY: all clean
