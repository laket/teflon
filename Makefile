USE_GPU := 0

DIR := ./

TARGET_NAME := buffer_ops.so

TF_INC := $(shell python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB := $(shell python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

# NDEBUG fixes a bug: https://github.com/tensorflow/tensorflow/issues/17316
FLAGS := -std=c++11 -shared -fPIC -I$(TF_INC) -I$(TF_INC)/external/nsync/public -L$(TF_LIB) -D_GLIBCXX_USE_CXX11_ABI=0 -O2 -DNDEBUG
CXX := g++
LDFLAGS := -ltensorflow_framework

SOURCES := $(DIR)/replay_buffer/*.cpp

ifeq ($(USE_GPU), 1)
    FLAGS += -DGOOGLE_CUDA=1
endif

all: $(TARGET_NAME)

$(TARGET_NAME):
	$(CXX) $(FLAGS) $(SOURCES) $(LDFLAGS) -o $@

clean:
	rm -rf $(TARGET_NAME)

remake: clean all
