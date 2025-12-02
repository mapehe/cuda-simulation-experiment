# ------------------------------------------------
# Generic Makefile for CUDA Projects (Enhanced for GTest)
# ------------------------------------------------

# Compiler settings
NVCC          = nvcc
NVCC_FLAGS    = -O3 -std=c++20 --extended-lambda
CXX_FLAGS     = -O3 -std=c++20
LIBS          = -lcufft

# [NEW] Google Test Libraries
TEST_LIBS     = -lgtest -lgtest_main -lpthread

# Dependency Generation Flags
DEP_FLAGS     = -MMD

# Project definitions
TARGET        = main
TEST_TARGET   = run_tests
SRC_DIR       = src
TEST_SRC_DIR  = src-test
OBJ_DIR       = obj
TEST_OBJ_DIR  = obj-test
BIN_DIR       = bin
INC_DIR       = include

# ------------------------------------------------
# Source & Object Discovery
# ------------------------------------------------

# 1. Application Sources
CU_SOURCES    = $(shell find $(SRC_DIR) -name "*.cu")
CPP_SOURCES   = $(shell find $(SRC_DIR) -name "*.cpp")

# 2. Test Sources [NEW]
TEST_CU_SOURCES  = $(shell find $(TEST_SRC_DIR) -name "*.cu")
TEST_CPP_SOURCES = $(shell find $(TEST_SRC_DIR) -name "*.cpp")

# 3. Object Generation
CU_OBJECTS    = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(CU_SOURCES))
CPP_OBJECTS   = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(CPP_SOURCES))
OBJECTS       = $(CU_OBJECTS) $(CPP_OBJECTS)

# [NEW] Test Objects
TEST_CU_OBJECTS  = $(patsubst $(TEST_SRC_DIR)/%.cu, $(TEST_OBJ_DIR)/%.o, $(TEST_CU_SOURCES))
TEST_CPP_OBJECTS = $(patsubst $(TEST_SRC_DIR)/%.cpp, $(TEST_OBJ_DIR)/%.o, $(TEST_CPP_SOURCES))
TEST_OBJECTS     = $(TEST_CU_OBJECTS) $(TEST_CPP_OBJECTS)

# 4. Dependency Files
DEPS          = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.d, $(CU_SOURCES)) \
                $(patsubst $(TEST_SRC_DIR)/%.cu, $(TEST_OBJ_DIR)/%.d, $(TEST_CU_SOURCES))

# ------------------------------------------------
# [IMPORTANT] Filter out main.o for testing
# ------------------------------------------------
# We need to link the core logic (wavefunction.o) with the tests, 
# BUT we must exclude the application's entry point (main.o) 
# because GTest provides its own main().
#
# ASSUMPTION: Your app's main function is in src/main.cu or src/main.cpp
APP_MAIN_OBJ  = $(OBJ_DIR)/main.o
# Create a list of objects intended for the library (Shared code)
LIB_OBJECTS   = $(filter-out $(APP_MAIN_OBJ), $(OBJECTS))

# ------------------------------------------------
# Build Rules
# ------------------------------------------------

all: directories $(BIN_DIR)/$(TARGET)

# [NEW] Test Target
test: directories $(BIN_DIR)/$(TEST_TARGET)

# Link Application
$(BIN_DIR)/$(TARGET): $(OBJECTS)
	@echo "Linking Application..."
	$(NVCC) $(OBJECTS) $(LIBS) -o $@
	@echo "Build complete: $@"

# [NEW] Link Tests
# Links: (Core Logic Objects without main) + (Test Objects) + (GTest Libs)
$(BIN_DIR)/$(TEST_TARGET): $(LIB_OBJECTS) $(TEST_OBJECTS)
	@echo "Linking Tests..."
	$(NVCC) $(LIB_OBJECTS) $(TEST_OBJECTS) $(LIBS) $(TEST_LIBS) -o $@
	@echo "Test Build complete: $@"

# Compile App CUDA
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@echo "Compiling CUDA $<..."
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) $(DEP_FLAGS) -I$(INC_DIR) -c $< -o $@

# Compile App C++
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Compiling C++ $<..."
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) $(DEP_FLAGS) -I$(INC_DIR) -c $< -o $@

# [NEW] Compile Test CUDA
$(TEST_OBJ_DIR)/%.o: $(TEST_SRC_DIR)/%.cu
	@echo "Compiling Test CUDA $<..."
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) $(DEP_FLAGS) -I$(INC_DIR) -c $< -o $@

# [NEW] Compile Test C++
$(TEST_OBJ_DIR)/%.o: $(TEST_SRC_DIR)/%.cpp
	@echo "Compiling Test C++ $<..."
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) $(DEP_FLAGS) -I$(INC_DIR) -c $< -o $@

directories:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(TEST_OBJ_DIR)
	@mkdir -p $(BIN_DIR)

-include $(DEPS)

clean:
	@echo "Cleaning up..."
	rm -rf $(OBJ_DIR) $(TEST_OBJ_DIR) $(BIN_DIR)

run: all
	@echo "Running App..."
	@./$(BIN_DIR)/$(TARGET)

# [NEW] Run Tests
run-test: test
	@echo "Running Tests..."
	@./$(BIN_DIR)/$(TEST_TARGET)

.PHONY: all clean run test run-test directories
