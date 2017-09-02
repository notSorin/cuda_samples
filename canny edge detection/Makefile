CC   = gcc
NVCC = nvcc 

EXE   = imgprocess 

SOURCES    = main.c
CU_SOURCES = kernel.cu


OBJS    = $(SOURCES:.c=.o) $(CU_SOURCES:.cu=.o)

CFLAGS     = -O3
NVCFLAGS   = $(CFLAGS) -Wno-deprecated-gpu-targets

LIBS = -L/usr/local/cuda/lib64/ -lcudart -lm 

SOURCEDIR = .

$(EXE) :$(OBJS) 
	$(CC) $(CFLAGS)  -o $@ $? -I. $(LIBS)

$(SOURCEDIR)/%.o : $(SOURCEDIR)/%.cu
	$(NVCC) $(NVCFLAGS) -c -o $@ $< -I.

$(SOURCEDIR)/%.o : $(SOURCEDIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $< -I.
clean:
	rm -f $(OBJS) $(EXE)
