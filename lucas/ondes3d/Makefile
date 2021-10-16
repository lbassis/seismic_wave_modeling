#CFLAGS          =   $(TESTFLAGS)  $(OPTI)  $(MODEL)
STARPU_VERSION=1.3

CFLAGS += -g
CFLAGS += $(shell pkg-config --cflags starpu-$(STARPU_VERSION))
LDLIBS += $(shell pkg-config --libs starpu-$(STARPU_VERSION))
LDLIBS += -Wl,-rpath -Wl,$(shell pkg-config --variable=libdir starpu-$(STARPU_VERSION))

PREFIX =./ondes3d
OBJS = main.o IO.o alloAndInit.o alloAndInit_LayerModel.o new_nrutil.o memory_dump.o main_loop.o \
        new_compute_seismoment.o new_compute_intermediates.o new_compute_stress.o new_compute_velo.o
HEADERS=include/struct.h include/options.h include/inlineFunctions.h


all: $(PREFIX)
$(PREFIX): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm $(LDLIBS)

%.o: src/%.c
	$(CC) $(CFLAGS) -c $< $(LDLIBS)

clean:
	rm -f *.o  *~ ./ondes3d

valgrind:
	valgrind ./ondes3d 33

valgrind-gdb:
	valgrind --vgdb=yes --vgdb-error=0 ./ondes3d 33 &
	(echo "target remote | vgdb"; cat) | gdb
