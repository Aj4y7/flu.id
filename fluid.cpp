#include "fluid.h"
#include <utility>

void set_bnd(int N, int b, std::vector<float>& x){
    for(int i = 1; i <= N; ++i){
        x[ix(0, i)] = b == 1 ? -x[ix(1, i)] : x[ix(1, i)];
        x[ix(N + 1, i)] = b == 1 ? -x[ix(N, i)] : x[ix(N, i)];
        x[ix(i, 0)] = b == 2 ? -x[ix(i, 1)] : x[ix(i, 1)];
        x[ix(i, N + 1)] = b == 2 ? -x[ix(i, N)] : x[ix(i, N)];
    }

    x[ix(0, 0)] = 0.5 * (x[ix(1, 0)] + x[ix(0, 1)]);
    x[ix(0, N + 1)] = 0.5 * (x[ix(1, N + 1)] + x[ix(0, N)]);
    x[ix(N + 1, 0)] = 0.5 * (x[ix(N, 0)] + x[ix(N + 1, 1)]);
    x[ix(N + 1, N + 1)] = 0.5 * (x[ix(N, N + 1)] + x[ix(N + 1, N)]);
}

void diffuse(int N, int b, std::vector<float>& x, std::vector<float>& x0, float diff, float dt){
    float a = dt * diff * N * N; 

    for(int k = 0; k <= 10; ++k){
        #pragma omp parallel for collapse(2)
        for(int i = 1; i <= N; ++i){
            for(int j = 1; j <= N; ++j){
                x[ix(i, j)] = (x0[ix(i, j)] + a * (x[ix(i - 1, j)] + x[ix(i + 1, j)] + x[ix(i, j - 1)] + x[ix(i, j + 1)])) / (1 + 4 * a);

            }
        }
    }
    
    set_bnd(N, b, x);
}

void vortConf(int N, std::vector<float>& vx, std::vector<float>& vy, float eps, float dt){
  std::vector<float> w((N + 2) * (N + 2), 0.f);

  #pragma omp parallel for collapse(2)
        for(int i = 1; i <= N; ++i){
    for(int j = 1; j <= N; ++j){
      w[ix(i, j)] = 0.5f * ((vy[ix(i + 1, j)] - vy[ix(i - 1, j)]) - (vx[ix(i, j + 1)] - vx[ix(i, j - 1)]));
    }
  }

  #pragma omp parallel for collapse(2)
  for(int i = 2; i < N; ++i){
    for(int j = 2; j < N; ++j){
      float dwX = 0.5f * (std::abs(w[ix(i + 1, j)]) - std::abs(w[ix(i - 1, j)]));
      float dwY = 0.5f * (std::abs(w[ix(i, j + 1)]) - std::abs(w[ix(i, j - 1)]));
      float len = std::sqrt(dwX * dwX + dwY * dwY) + 1e-6f;

      float Nx = dwX / len, Ny = dwY / len;
      float vort = w[ix(i, j)];

      float Fx = eps * Ny * vort;
      float Fy = -eps * Nx * vort;
      
      vx[ix(i, j)] += Fx * dt;
      vy[ix(i, j)] += Fy * dt;
    }
  }
}

void advect(int N, int b, std::vector<float>& density, std::vector<float>& densityPrev, std::vector<float>& vx, std::vector<float>& vy, float dt){
    dt *= N;
    #pragma omp parallel for collapse(2)
        for(int i = 1; i <= N; ++i){
        for(int j = 1; j <= N; ++j){
            float x = i - dt * vx[ix(i, j)], y = j - dt * vy[ix(i, j)];
            x = std::clamp(x, 0.5f, N + 0.5f);
            y = std::clamp(y, 0.5f, N + 0.5f);
            int i0 = floor(x);
            int i1 = i0 + 1;
            int j0 = floor(y);
            int j1 = j0 + 1;
            float s1 = x - i0, s0 = 1 - s1, t1 = y - j0, t0 = 1 - t1;

            density[ix(i, j)] = s0 * (t0 * densityPrev[ix(i0, j0)] + t1 * densityPrev[ix(i0, j1)]) + s1 * (t0 * densityPrev[ix(i1, j0)] + t1 * densityPrev[ix(i1, j1)]);
        }
    }
    set_bnd(N, b, density);
}

void densStep(int N, std::vector<float>& x, std::vector<float>& x0, std::vector<float>& vx, std::vector<float>& vy, float diff, float dt){
    std::swap(x, x0);
    diffuse(N, 0, x, x0, diff, dt);    
    
    std::swap(x, x0);
    advect(N, 0, x, x0, vx, vy, dt);
}

void project(int N, std::vector<float>& vx, std::vector<float>& vy, std::vector<float>& p, std::vector<float>& div){
    float h = 1.0 / (float)N;

    #pragma omp parallel for collapse(2)
        for(int i = 1; i <= N; ++i){
        for(int j = 1; j <= N; ++j){
            div[(ix(i, j))] = -0.5 * h * (vx[ix(i + 1, j)] - vx[ix(i - 1, j)] + vy[ix(i, j + 1)] - vy[ix(i, j - 1)]);
            p[ix(i, j)] = 0;
        }
    }

    for(int k = 0; k <= 10; ++k){
        #pragma omp parallel for collapse(2)
        for(int i = 1; i <= N; ++i){
            for(int j = 1; j <= N; ++j){
                p[ix(i, j)] = (div[ix(i, j)] + p[ix(i - 1, j)] + p[ix(i + 1, j)] + p[ix(i, j - 1)] + p[ix(i, j + 1)]) / 4;
            }
        }
        set_bnd(N, 0, p);
    }

    #pragma omp parallel for collapse(2)
        for(int i = 1; i <= N; ++i){
        for(int j = 1; j <= N; ++j){
            vx[ix(i, j)] -= 0.5 * (p[ix(i + 1, j)] - p[ix(i - 1, j)]) / h;
            vy[ix(i, j)] -= 0.5 * (p[ix(i, j + 1)] - p[ix(i, j - 1)]) / h;
        }
    }

    set_bnd(N, 1, vx); 
    set_bnd(N, 2, vy);
}

void velStep(int N, std::vector<float>& vx, std::vector<float>& vy, std::vector<float>& vx0, std::vector<float>& vy0, std::vector<float>& p, std::vector<float>& div, float visc, float dt){
   const float eps = 3.0f;
   std::swap(vx, vx0);
   std::swap(vy, vy0);

   diffuse(N, 1, vx, vx0, visc, dt); 
   diffuse(N, 2, vy, vy0, visc, dt); 
   
   vortConf(N, vx, vy, eps, dt);
   project(N, vx, vy, p, div);
   std::swap(vx, vx0);
   std::swap(vy, vy0);

   advect(N, 1, vx, vx0, vx0, vy0, dt);
   advect(N, 2, vy, vy0, vx0, vy0, dt);

   project(N, vx, vy, p, div);
}