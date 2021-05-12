#CFLAGS          =   $(TESTFLAGS)  $(OPTI)  $(MODEL) 
PREFIX =./ondes3d
OBJS = main.o IO.o nrutil.o #computeVeloAndSource.o computeStress.o computeIntermediates.o alloAndInit.o IO.o alloAndInit_LayerModel.o
HEADERS=include/struct.h include/options.h include/inlineFunctions.h


all: $(PREFIX)
$(PREFIX): $(OBJS)
	$(CC) -o $@ $^ -lm 

%.o: src/%.c
	$(CC) -c $<

clean:
	rm -f *.o  *~ ./ondes3d
