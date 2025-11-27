#ifndef SIMULATION_H
#define SIMULATION_H

#include "json.hpp"
#include "kernel/testKernel.h"
#include <fstream>
#include <iostream>

using json = nlohmann::json;

void run(json config);

#endif
