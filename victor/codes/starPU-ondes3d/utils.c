#include <stdio.h>
#include "utils.h"
#include "starpu.h"
//#define VERBOSE_GRAPH

task_info events[STARPU_NMAXWORKERS][NMAX_TASKS];
unsigned int nb_tasks_performed[STARPU_NMAXWORKERS];

char* colors[12] = {
  "Red",
  "Lime",
  "Aqua",
  "Brown",
  "Pink",
  "HotPink",
  "DeepPink",
  "MediumVioletRed",
  "Yellow",
  "Moccasin",
  "PeachPuff",
  "Khaki"
  };

char* names[12] = {
  "stress",
  "veloc",
  "source",
  "sismos",
  "str_xp",
  "str_xm",
  "str_yp",
  "str_ym",
  "vel_xp",
  "vel_xm",
  "vel_yp",
  "vel_ym"
  };
  
struct timespec starttime;
struct timespec endtime;

#define relative_ms(T) msecdur(starttime,T)
#define ratio(duration) (float)((duration) / msecdur(starttime,endtime))

int width = 16384;
int height = 500;
int margin = 50;
float interligne;
char name[64];
task_info* dependencies[20];

void init_drawing() {
	int i;
	gettime(starttime);
	for(i=0; i<STARPU_NMAXWORKERS; i++)
		nb_tasks_performed[i] = 0;
	return;
}

void register_event(unsigned int worker, unsigned int type, unsigned int bx, unsigned int by, unsigned int iter, struct timespec start, struct timespec end) {
	task_info *ti = &events[worker][nb_tasks_performed[worker]++];

	ti->type = type;
	ti->bx = bx;
	ti->by = by;
	ti->iter = iter;
	ti->start = relative_ms(start);
	ti->end = relative_ms(end);
	ti->dep = 0;
	ti->worker = worker;
}

void draw_timeline(FILE* out, char* name, float x1, float x2, float y1, float y2) {
	// ligne
	fprintf(out, "<line x1=\"%f\" y1=\"%f\" x2=\"%f\" y2=\"%f\" style=\"stroke:Black\"/>\n", x1, y1, x2, y2);
	// nom
	fprintf(out, "<text x=\"%f\" y=\"%f\">%s</text>\n", 0.f, y1, name);
	return;
}

void draw_task(FILE* out, task_info *ti, int numworker) {
	sprintf(name, "%s{%d,%d} it:%d dur:%.4f ms",names[ti->type], ti->bx, ti->by, ti->iter, ti->end-ti->start);

	float task_width = ratio((ti->end-ti->start))*(width-2*margin);
	float task_height = interligne/2.f;
	float posx = margin + ratio(ti->start)*(width-2*margin);
	float posy = margin + numworker*interligne - interligne/4.f;
	fprintf(out, "<rect x=\"%f\" y=\"%f\" width=\"%f\" height=\"%f\"", posx, posy, task_width, task_height);
	fprintf(out, " style=\" fill:%s; stroke:%s\" onmousemove=\"ShowTooltip(evt, '%s')\" onmouseout=\"HideTooltip()\"/>\n", colors[ti->type], "Black", name);
	return;
}


task_info* find_task(tasktype type, unsigned int bx, unsigned int by, int iter) {

	int iworker, itask;
	task_info* ti;

	for (iworker=0; iworker<STARPU_NMAXWORKERS; iworker++) {
		for (itask=0; itask<nb_tasks_performed[iworker]; itask++) {
			ti = &events[iworker][itask];
			if (ti->type == type && ti->bx == bx && ti->by == by && ti->iter == iter) return ti;
		}
	}
	return NULL;
}

int find_deps(task_info *ti) {
	int nbdep = 0;
	int numworker;
	task_info* t;

	switch(ti->type) {
		case _STRESS :	if (t=find_task(_VELOC, ti->bx, ti->by, ti->iter)) dependencies[nbdep++]=t;
						if (t=find_task(_V_XP, ti->bx-1, ti->by, ti->iter)) dependencies[nbdep++]=t;
						if (t=find_task(_V_XM, ti->bx+1, ti->by, ti->iter)) dependencies[nbdep++]=t;
						if (t=find_task(_V_YP, ti->bx, ti->by-1, ti->iter)) dependencies[nbdep++]=t;
						if (t=find_task(_V_YM, ti->bx, ti->by+1, ti->iter)) dependencies[nbdep++]=t;
						break;
		case _VELOC :	if (t=find_task(_STRESS, ti->bx, ti->by, ti->iter-1)) dependencies[nbdep++] = t;
						if (t=find_task(_S_XP, ti->bx-1, ti->by, ti->iter-1)) dependencies[nbdep++] = t;
						if (t=find_task(_S_XM, ti->bx+1, ti->by, ti->iter-1)) dependencies[nbdep++] = t;
						if (t=find_task(_S_YP, ti->bx, ti->by-1, ti->iter-1)) dependencies[nbdep++] = t;
						if (t=find_task(_S_YM, ti->bx, ti->by+1, ti->iter-1)) dependencies[nbdep++] = t;
						break;
		case _S_XP :	
		case _S_XM :	
		case _S_YP :	
		case _S_YM :	if (t=find_task(_STRESS, ti->bx, ti->by, ti->iter)) dependencies[nbdep++] = t;
						break;
		case _V_XP :	
		case _V_XM :	
		case _V_YP :	
		case _V_YM :	if (t=find_task(_VELOC, ti->bx, ti->by, ti->iter)) dependencies[nbdep++] = t;
						break;
//		case default : ;	
//						break;
	}
	return nbdep;
}

void draw_dep(FILE* svg, task_info* ti) {
	int i;
	float x1, x2, y1, y2;
	int nbdep = find_deps(ti);

#ifdef VERBOSE_GRAPH	
	printf("%s %d,%d / %d", names[ti->type], ti->bx, ti->by, ti->iter);
	for (i=0; i<nbdep; i++) {
			printf("\t-> %s %d,%d / %d\n", names[dependencies[i]->type], dependencies[i]->bx, dependencies[i]->by, dependencies[i]->iter);	
	}
	printf("\n");
#endif

	task_info* dep;

	for (i=0; i<nbdep; i++) {
		dep = dependencies[i];
		x1 = margin + ratio(dep->end)*(width-2*margin);
		y1 = margin + dep->worker*interligne;
		x2 = margin + ratio(ti->start)*(width-2*margin);
		y2 = margin + ti->worker*interligne;
		fprintf(svg, "<line class=\"dep\" x1=\"%f\" y1=\"%f\" x2=\"%f\" y2=\"%f\" style=\"stroke:Blue\"/>\n", x1, y1, x2, y2);
	}
}

void draw_svg(char* svgfile) {
	int iworker, itask, numworker;
	int nbworker = 0;

	task_info *ti;
	FILE* svg;

	float x1, x2, y1, y2;
	int i;

	gettime(endtime);
	// create file, write header
	svg = fopen(svgfile, "w+");
	fprintf(svg, "<?xml version=\"1.0\" standalone=\"no\"?>\n");
	fprintf(svg, "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\" \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n");
	fprintf(svg, "<svg width=\"%d\" height=\"%d\" viewBox=\"0 0 %d %d\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/2000/xlink\" version=\"1.1\" onload=\"init(evt)\">\n", width, height, width, height);
	fprintf(svg, "   <style>\n");
	fprintf(svg, "     .caption{\n");
	fprintf(svg, " 	font-size: 14px;\n");
	fprintf(svg, " 	font-family: Georgia, serif;\n");
	fprintf(svg, "     }\n");
	fprintf(svg, "     .tooltip{\n");
	fprintf(svg, " 	font-size: 12px;\n");
	fprintf(svg, "     }\n");
	fprintf(svg, "     .tooltip_bg{\n");
	fprintf(svg, " 	fill: white;\n");
	fprintf(svg, " 	stroke: black;\n");
	fprintf(svg, " 	stroke-width: 1;\n");
	fprintf(svg, " 	opacity: 0.85;\n");
	fprintf(svg, "     }\n");
	fprintf( svg, "     .dep{\n");
	fprintf( svg, "    pointer-events: none;\n");
	fprintf( svg, "	 }\n");
	fprintf(svg, "   </style>\n");
	fprintf(svg, "   <script type=\"text/ecmascript\">\n");
	fprintf(svg, "     <![CDATA[\n");
	fprintf(svg, " \n");
	fprintf(svg, " 	function init(evt)\n");
	fprintf(svg, " 	{\n");
	fprintf(svg, " 	    if ( window.svgDocument == null )\n");
	fprintf(svg, " 	    {\n");
	fprintf(svg, " 		svgDocument = evt.target.ownerDocument;\n");
	fprintf(svg, " 	    }\n");
	fprintf(svg, " \n");
	fprintf(svg, " 	    tooltip = svgDocument.getElementById('tooltip');\n");
	fprintf(svg, " 	    tooltip_bg = svgDocument.getElementById('tooltip_bg');\n");
	fprintf(svg, "  	}\n");
	fprintf(svg, "\n");
	fprintf(svg, "  	function getScrollXY() {\n");
	fprintf(svg, "      var scrOfX = 0, scrOfY = 0;\n");
	fprintf(svg, "      if( typeof( window.pageYOffset ) == 'number' ) {\n");
	fprintf(svg, "        //Netscape compliant\n");
	fprintf(svg, "        scrOfY = window.pageYOffset;\n");
	fprintf(svg, "        scrOfX = window.pageXOffset;\n");
	fprintf(svg, "      } else if( document.body && ( document.body.scrollLeft || document.body.scrollTop ) ) {\n");
	fprintf(svg, "        //DOM compliant\n");
	fprintf(svg, "        scrOfY = document.body.scrollTop;\n");
	fprintf(svg, "        scrOfX = document.body.scrollLeft;\n");
	fprintf(svg, "      } else if( document.documentElement && ( document.documentElement.scrollLeft || document.documentElement.scrollTop ) ) {\n");
	fprintf(svg, "        //IE6 standards compliant mode\n");
	fprintf(svg, "        scrOfY = document.documentElement.scrollTop;\n");
	fprintf(svg, "        scrOfX = document.documentElement.scrollLeft;\n");
	fprintf(svg, "      }\n");
	fprintf(svg, "      return [ scrOfX, scrOfY ];\n");
	fprintf(svg, "    }\n");
	fprintf(svg, " \n");
	fprintf(svg, " 	function ShowTooltip(evt, mouseovertext)\n");
	fprintf(svg, " 	{	\n");
	fprintf(svg, " 	 	var scrollpos = getScrollXY();\n");
	fprintf(svg, " 	    tooltip.setAttributeNS(null,\"x\",evt.clientX+scrollpos[0]+11);\n");
	fprintf(svg, " 	    tooltip.setAttributeNS(null,\"y\",evt.clientY+scrollpos[1]+27);\n");
	fprintf(svg, " 	    tooltip.firstChild.data = mouseovertext;\n");
	fprintf(svg, " 	    tooltip.setAttributeNS(null,\"visibility\",\"visible\");\n");
	fprintf(svg, " \n");
	fprintf(svg, " 	    length = tooltip.getComputedTextLength();\n");
	fprintf(svg, " 	    tooltip_bg.setAttributeNS(null,\"width\",length+8);\n");
	fprintf(svg, " 	    tooltip_bg.setAttributeNS(null,\"x\",evt.clientX+scrollpos[0]+8);\n");
	fprintf(svg, " 	    tooltip_bg.setAttributeNS(null,\"y\",evt.clientY+scrollpos[1]+14);\n");
	fprintf(svg, " 	    tooltip_bg.setAttributeNS(null,\"visibility\",\"visibile\");\n");
	fprintf(svg, " 	}\n");
	fprintf(svg, " \n");
	fprintf(svg, " 	function HideTooltip(evt)\n");
	fprintf(svg, " 	{\n");
	fprintf(svg, " 	    tooltip.setAttributeNS(null,\"visibility\",\"hidden\");\n");
	fprintf(svg, " 	    tooltip_bg.setAttributeNS(null,\"visibility\",\"hidden\");\n");
	fprintf(svg, " 	}\n");
	fprintf(svg, " \n");
	fprintf(svg, "     ]]>\n");
	fprintf(svg, "   </script>\n");


	for (iworker=0; iworker<STARPU_NMAXWORKERS; iworker++) {
		if (nb_tasks_performed[iworker]) {
			nbworker = iworker+1;
		}
	}
	// first draw timelines
	interligne = (nbworker>1)?(height - 2*margin) / (nbworker-1):margin/2.f;
	x1 = margin;
	x2 = width - margin;
	for (iworker=0; iworker<STARPU_NMAXWORKERS; iworker++) {
		if (nb_tasks_performed[iworker]) {
			y1 = y2 = margin + iworker*interligne;
			starpu_worker_get_name(iworker, name, sizeof(name));
			draw_timeline(svg, name, x1, x2, y1, y2);
		}
	}

	// draw dependences
	for (iworker=0; iworker<STARPU_NMAXWORKERS; iworker++) {
		for (itask=0; itask<nb_tasks_performed[iworker]; itask++) {
			ti = &events[iworker][itask];				
			draw_dep(svg, ti);
		}
	}

	// then draw tasks
	numworker=0;
	for (iworker=0; iworker<STARPU_NMAXWORKERS; iworker++) {
		for (itask=0; itask<nb_tasks_performed[iworker]; itask++) {
			draw_task(svg, &events[iworker][itask], iworker);
		}
	}

	// Finalize svg & close file
	fprintf(svg, "<rect class=\"tooltip_bg\" id=\"tooltip_bg\" x=\"0\" y=\"0\" rx=\"4\" ry=\"4\" width=\"55\" height=\"17\" visibility=\"hidden\"/>");
	fprintf(svg, "<text class=\"tooltip\" id=\"tooltip\" x=\"0\" y=\"0\" visibility=\"hidden\">Tooltip</text></svg>\n");
	fclose(svg);
	return;
}
