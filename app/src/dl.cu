#include "MODEL/DL.hpp"

int main(void)
{
  MODEL::DL dl(64);
  dl.train(200,0.00999999977648258209F, 32);
  
  
  return 0;
}
