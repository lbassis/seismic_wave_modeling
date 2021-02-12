#ifndef __ONDES3D_COMMON_H__
#define __ONDES3D_COMMON_H__
#define DELTA 0
#define NPOWER 2.0

#define MASK_FIRST_X 1
#define MASK_LAST_X 2
#define MASK_FIRST_Y 4
#define MASK_LAST_Y 8
#define DUMMY_VALUE 100

#define MIN(a,b)  ((a)<(b)?(a):(b))
#define MAX(a,b)  ((a)>(b)?(a):(b))

// stencil
#define K 2

#define ALIGN 16
#define BLOCKSIZE 32

typedef enum
{
  XP = 0,
  XM = 1,
  YP = 2,
  YM = 3
} direction;

typedef enum
{
  FROM_BUF = 0,
  TO_BUF = 1
} sens;
#endif
