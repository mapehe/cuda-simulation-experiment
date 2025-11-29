# ------------------------------------------------
# Generic Makefile for CUDA Projects
# ------------------------------------------------

# Compiler settings
NVCC          = nvcc
# Add architecture flags here (e.g., -arch=sm_75) if you know your GPU
NVCC_FLAGS    = -O3 -std=c++20
# NVCC_FLAGS += -arch=sm_70 
LIBS          = -lcufft

# **Dependency Generation Flags**
# -M: Tells nvcc/gcc/g++ to output a rule suitable for make describing the dependencies of the source file.
# -MMD: Same as -M but omits system headers, and outputs the rule to a .d file.
DEP_FLAGS     = -MMD

# Project definitions
TARGET        = main
SRC_DIR       = src
OBJ_DIR       = obj
BIN_DIR       = bin
INC_DIR       = include

SOURCES       = $(shell find $(SRC_DIR) -name "*.cu")
OBJECTS       = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SOURCES))
DEPS          = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.d, $(SOURCES))

# ------------------------------------------------
# Build Rules
# ------------------------------------------------

# Default target: build the executable
all: directories $(BIN_DIR)/$(TARGET)

# Rule to link object files into the final executable
$(BIN_DIR)/$(TARGET): $(OBJECTS)
	@echo "Linking..."
	$(NVCC) $(OBJECTS) $(LIBS) -o $@
	@echo "Build complete: $@"

# Rule to compile .cu files into .o files
# **Dependency generation is added here via $(DEP_FLAGS)**
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@echo "Compiling $<..."
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) $(DEP_FLAGS) -I$(INC_DIR) -c $< -o $@

# Create specific directories if they don't exist
directories:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(BIN_DIR)

# **Crucial step: Include the generated dependency files.**
# This instructs Make to look inside each .d file to find the header dependencies
# and recompile the corresponding .o file if any of them are newer.
-include $(DEPS)

# Clean up build artifacts
# **Now also cleans up the dependency files (.d)**
clean:
	@echo "Cleaning up..."
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# Run the program
run: all
	@echo "Running $(TARGET)..."
	@./$(BIN_DIR)/$(TARGET)

# Phony targets help avoid conflicts with files named 'clean' or 'all'
.PHONY: all clean run directories
