#ifndef UTILS_H
#define UTILS_H

#include "stencil.h"

#define NMAX_TASKS 10000

typedef enum {
	_STRESS = 0,
	_VELOC = 1,
	_SOURCE = 2,
	_SISMO = 3,
	_S_XP = 4,
	_S_XM = 5,
	_S_YP = 6,
	_S_YM = 7,
	_V_XP = 8,
	_V_XM = 9,
	_V_YP = 10,
	_V_YM = 11
} tasktype;

typedef struct {
	// task type
	tasktype type;
	// start & end time
	float start;
	float end;
	// block
	unsigned int bx;
	unsigned int by;
	// iteration
	unsigned int iter;
	unsigned int dep;
	int worker;
} task_info;

void init_drawing();
void register_event(unsigned int worker, tasktype type, unsigned int bx, unsigned int by, unsigned int iter, struct timespec start, struct timespec end);
void draw_svg(char* svgfile);
void draw_timeline(FILE* out, char* name, float x1, float x2, float y1, float y2);
void draw_task(FILE* out, task_info *ti, int numworker);
#endif
