#WISDM:
#gcc -D K=3 -D DT=1 -D NUM_FEATURES=43 -D SCENARIO_FEATURES=1 -D READ=4 -D TRAIN_PATH=\"..\\data\\wisdm\\train2_norm.dat\" -D TEST_PATH=\"..\\data\\wisdm\\test2_notnorm.dat\" -D DIMEM=0 -D VERIFY=0 -D STREAMING=1 -O2 -Wall -Wextra -std=gnu99 src\knn.c src\utils.c src\timer.c src\io.c src\features.c src\main.c -I..\data\wisdm -o harknn -lm
#
#PAMAP2:
#gcc -D K=3 -D DT=1 NUM_FEATURES=90 -D SCENARIO_FEATURES=2 -D READ=4 -D TRAIN_PATH=\"..\\data\\pamap2\\train2_norm.dat\" -D TEST_PATH=\"..\\data\\pamap2\\test2_notnorm.dat\" -D DIMEM=0 -D VERIFY=0 -D STREAMING=1 -O2 -Wall -Wextra -std=gnu99 src\knn.c src\utils.c src\timer.c src\io.c src\features.c src\main.c -I..\data\wisdm -o harknn -lm
#

# the following definitions are passed through gcc parameters

#define DT 2 or 1
#define READ 4 or 2
#define SPECIALIZED 1 or 0
#define MATH_TYPE 1
#define DIST_METHOD 1
#define USE_SQRT 1 or 0
#define DIMEM 0
#define VERIFY 0
#define STREAMING 1
#define K 3
#define TIMING 0
#define NORMALIZE 1
#define SCENARIO_FEATURES 1 or 2
#define NUM_FEATURES 43 or 90

# the path to the datasets folder (absolute, so the binary can run from any directory)
DATA_DIR ?= $(abspath $(CURDIR)/../data)

#variable can be overridden: wisdm or papmap2
SCENARIO ?= wisdm
K ?= 3
# 2: float; 1: double
DT ?= 1
SPECIALIZED ?= 1
USE_SQRT ?= 1

READ ?= 4

#compiler optimizations
OPT ?= -O2

# to work in Windows and Linux
ifeq ($(OS),Windows_NT)
	RM = del
	RMFLAGS = /Q
	EXE = .exe
	PATH_SEP = \\
else
	RM = rm
	RMFLAGS = -f
	EXE =
	PATH_SEP = /
endif

#scenario options
ifeq ($(SCENARIO),wisdm)
HARKNNDATASET = $(DATA_DIR)$(PATH_SEP)wisdm
NUM_FEATURES = 43
SCENARIO_FEATURES = 1
else
HARKNNDATASET = $(DATA_DIR)$(PATH_SEP)pamap2
NUM_FEATURES = 90
SCENARIO_FEATURES = 2
endif
INCLUDES = $(HARKNNDATASET)

echo1:
	@echo $(DATA_DIR)
echo2:
	@echo $(HARKNNDATASET)
echo3:
	@echo $(INCLUDES)
	
ifeq ($(READ),3)
TRAIN_PATH = '$(HARKNNDATASET)$(PATH_SEP)train2_notnorm.dat'
TEST_PATH = '$(HARKNNDATASET)$(PATH_SEP)test2_notnorm.dat'
else ifeq ($(READ),4)
TRAIN_PATH = '$(HARKNNDATASET)$(PATH_SEP)train2_norm.dat'
TEST_PATH = '$(HARKNNDATASET)$(PATH_SEP)test2_notnorm.dat'
endif

SRC_DIR = src
OBJ_DIR = build
BIN_DIR = bin
RUN_DIR = run
CURRENT_DIR = .$(PATH_SEP)
PROGNAME = harknn$(EXE)
TARGET = $(BIN_DIR)$(PATH_SEP)$(PROGNAME)

CC = gcc
CFLAGS = $(OPT) \
	-I$(INCLUDES) \
	-Wall -Wextra -std=gnu99 \
	-D K=$(K) \
	-D DT=$(DT) \
	-D SPECIALIZED=$(SPECIALIZED) \
	-D MATH_TYPE=1 \
	-D DIST_METHOD=1 \
	-D USE_SQRT=$(USE_SQRT) \
	-D NUM_FEATURES=$(NUM_FEATURES) \
	-D SCENARIO_FEATURES=$(SCENARIO_FEATURES) \
	-D READ=$(READ) -D DIMEM=0 -D VERIFY=0 -D STREAMING=1 -D NORMALIZE=1 \
	-D TRAIN_PATH=\"$(TRAIN_PATH)\" \
	-D TEST_PATH=\"$(TEST_PATH)\"

LDFLAGS = -lm

PROFILING_FLAGS = -g -pg
PROFILING = gprof -b

#SRCS = knn.c utils.c timer.c io.c features.c main.c
SRCS = 	$(SRC_DIR)$(PATH_SEP)knn.c $(SRC_DIR)$(PATH_SEP)utils.c \
		$(SRC_DIR)$(PATH_SEP)timer.c $(SRC_DIR)$(PATH_SEP)io.c \
		$(SRC_DIR)$(PATH_SEP)features.c $(SRC_DIR)$(PATH_SEP)main.c
#OBJS = $(SRCS:.c=.o)
OBJS = $(patsubst $(SRC_DIR)$(PATH_SEP)%.c, $(OBJ_DIR)$(PATH_SEP)%.o, $(SRCS))

.PHONY: all clean cleanall run comprofiling runprofiling

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)$(PATH_SEP)%.o: $(SRC_DIR)$(PATH_SEP)%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) $(RMFLAG) $(OBJS)

cleanall: clean
	$(RM) $(RMFLAG) $(TARGET) $(BIN_DIR)$(PATH_SEP)gmon.out

comp: 
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

run: $(TARGET)
	$(CURRENT_DIR)$(TARGET)
	
comprofiling: 
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS) $(PROFILING_FLAGS)

runprofiling: comprofiling
	cd $(RUN_DIR) && \
	..$(PATH_SEP)$(BIN_DIR)$(PATH_SEP)$(PROGNAME) && \
	$(PROFILING) ..$(PATH_SEP)$(BIN_DIR)$(PATH_SEP)$(PROGNAME)