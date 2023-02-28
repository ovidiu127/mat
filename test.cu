#include <stdio.h>
#include <time.h>
#include <assert.h>

#include "matrix.h"

int main(){
  matrix<float>a(1024,1024),b(1024,1024),c;
  a.rinit();
  b.rinit();

  clock_t beg=clock(),end;

  dot(a,b,c);
  end=clock();
  printf("G: %f ms\n",(float)(end-beg)/((float)CLOCKS_PER_SEC/1000));

  beg=clock();
  check(a,b,c);
  end=clock();
  printf("C: %f ms\n",(float)(end-beg)/((float)CLOCKS_PER_SEC/1000));
  return 0;
}