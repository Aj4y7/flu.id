#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#define ix(i, j) ((i) + (N + 2) * (j))

void set_bnd(int N, int b, std::vector<float>& x);
void diffuse(int N, int b, std::vector<float>& x, std::vector<float>& x0, float diff, float dt);
void vortConf(int N, std::vector<float>& vx, std::vector<float>& vy, float eps, float dt);
void advect(int N, int b, std::vector<float>& density, std::vector<float>& densityPrev, std::vector<float>& vx, std::vector<float>& vy, float dt);
void densStep(int N, std::vector<float>& x, std::vector<float>& x0, std::vector<float>& vx, std::vector<float>& vy, float diff, float dt);
void project(int N, std::vector<float>& vx, std::vector<float>& vy, std::vector<float>& p, std::vector<float>& div);
void velStep(int N, std::vector<float>& vx, std::vector<float>& vy, std::vector<float>& vx0, std::vector<float>& vy0, std::vector<float>& p, std::vector<float>& div, float visc, float dt);
