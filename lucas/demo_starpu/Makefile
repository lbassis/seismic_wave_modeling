STARPU_VERSION=1.3
CPPFLAGS += $(shell pkg-config --cflags starpu-$(STARPU_VERSION))
LDLIBS += $(shell pkg-config --libs starpu-$(STARPU_VERSION))
LDLIBS += -Wl,-rpath -Wl,$(shell pkg-config --variable=libdir starpu-$(STARPU_VERSION))

%.o: %.c
	gcc $(CPPFLAGS) $< -c -o $@ $(LDLIBS)

scal: scaling.o
	gcc scaling.o $(LDLIBS) -o demo

clean:
	rm -f *.o  *~
