# High-Performance Fluid Simulation

![Fluid Simulation Demo](sim.gif)

A real-time, grid-based 2D fluid simulation written in C++. This project is heavily influenced by Jos Stam's seminal paper, *Real-Time Fluid Dynamics for Games* (GDC 2003), which forms the mathematical backbone of the entire engine. My main goal was to understand Stam's unconditionally stable solver for incompressible fluids, and then engineer the application to handle massive, high-resolution grids without dropping below 60 FPS.

## How it Works

**The Physics**
Following Stam's methodology, the engine is built on an Eulerian grid. It perfectly solves density and velocity fields using his famous semi-Lagrangian advection scheme alongside Gauss-Seidel relaxation for the mass-conserving projection step. To build on top of that foundation, I also implemented vorticity confinement, which adds necessary swirl and turbulence back into the fluid so it doesn't become artificially viscous over time.

**Performance & Architecture**
Getting a high-resolution grid (e.g., 256x256) to run smoothly on a single machine required a few key optimizations:
- **Multithreading:** Fluid grid math is highly parallelizable. I used OpenMP to divide the heavy 2D array calculations simultaneously across all available CPU cores.
- **Hardware Rendering:** Instead of drawing pixels one-by-one on the CPU, the engine batches the grid geometry into an `sf::VertexArray` to send directly to the GPU.
- **Color Look-Up Tables (LUT):** To give the fluid a smooth, scientific appearance, I used a Viridis perceptual colormap. Since calculating these colors on the fly is computationally expensive, I precompute them into an invisible 1D texture (the LUT). The CPU just calculates the fluid's density, and the GPU automatically snaps it to the right color.

## Building the Project

Compiler optimizations (`-O3`, `-ffast-math`, and OpenMP multithreading) are already hardcoded into the CMake configuration, meaning a standard build will automatically compile for maximum performance.

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
./fluid
```

## Controls
- **Click and drag:** Injects fluid dye and velocity into the field. The amount injected is scaled dynamically by how fast you move your mouse.

## References
- *Real-Time Fluid Dynamics for Games*, Jos Stam (GDC 2003)
