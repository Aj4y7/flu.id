# flu.id

<p align="center">
  <img src="sim.gif" width="400" alt="Fluid Simulation Demo">
</p>

Real‑time 2D fluid simulation on a grid, written in C++. The solver is built on Jos Stam's 2003 paper -- stable, well‑understood, and a solid starting point. The main challenge was making it scale. With some targeted optimizations, it runs 256×256 grids at a consistent 60 FPS on a single machine.

## How it Works

**Physics**  
The engine uses an Eulerian grid with Stam's semi‑Lagrangian advection -- unconditionally stable, so no timestep restrictions. Projection is handled by Gauss‑Seidel relaxation: simple, fast, and converges well. I also added vorticity confinement to counteract the artificial damping that naturally occurs, keeping the fluid lively and swirling instead of turning into sludge.

**Performance (the real engineering)**  
To push a 256×256 grid at 60 Hz, every unnecessary cost had to go:

- **OpenMP** – Grid operations are embarrassingly parallel, so I split the 2D array work across all CPU cores.
- **GPU batching** – Instead of rendering pixel by pixel on the CPU, I batch the grid into an `sf::VertexArray` and send it directly to the GPU.
- **Precomputed LUT** – The Viridis colormap gives a clean, perceptual look, but computing RGB per cell per frame is wasteful. So I pre‑baked the map into a 1D texture. The CPU only tracks density; the GPU does the color mapping at zero extra cost.

## Building

The CMake configuration already sets `-O3`, `-ffast-math`, and enables OpenMP -- so a standard build will compile with performance in mind.

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
./fluid
```

## Controls

- **Click and drag** – injects dye and velocity. The injection rate scales with mouse speed, which feels natural.

## References

- *Real‑Time Fluid Dynamics for Games*, Jos Stam (GDC 2003)
