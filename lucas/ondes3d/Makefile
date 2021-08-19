#CFLAGS          =   $(TESTFLAGS)  $(OPTI)  $(MODEL)
STARPU_VERSION=1.3

CFLAGS = -g
CFLAGS += $(shell pkg-config --cflags starpu-$(STARPU_VERSION))
LDLIBS += $(shell pkg-config --libs starpu-$(STARPU_VERSION))
LDLIBS += -Wl,-rpath -Wl,$(shell pkg-config --variable=libdir starpu-$(STARPU_VERSION))

PREFIX =./ondes3d
OBJS = main.o IO.o alloAndInit.o alloAndInit_LayerModel.o nrutil.o computeVeloAndSource.o computeIntermediates.o codelets.o computeStress.o
HEADERS=include/struct.h include/options.h include/inlineFunctions.h


all: $(PREFIX)
$(PREFIX): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm $(LDLIBS)

%.o: src/%.c
	$(CC) $(CFLAGS) -c $< $(LDLIBS)

clean:
	rm -f *.o  *~ ./ondes3d
