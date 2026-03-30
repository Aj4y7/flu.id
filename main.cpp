#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <SFML/Graphics/Image.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
#include "fluid.h"
#define ix(i, j) ((i) + (N + 2) * (j))
const float exposure = 2.5f;
const int N = 256;
const int CELL = 3;

sf::Color viridis(float t){
    t = std::clamp(t, 0.0f, 1.0f);

    struct Stop{float t, r, g, b;};
    static const Stop stops[] = {
        {0.00f, 68, 1, 84},
        {0.25f, 59, 82, 139},
        {0.50f, 33, 145, 140},
        {0.75f, 94, 201, 98},
        {1.00f, 253, 231, 37}
    };

    for(int k = 0; k < 4; ++k){
        if(t <= stops[k + 1].t){
            float u = (t - stops[k].t) / (stops[k + 1].t - stops[k].t);
            auto lerp = [u](float a, float b){
                return a + (b - a) * u; 
            };

            return sf::Color(
                static_cast<unsigned char>(lerp(stops[k].r, stops[k + 1].r)),
                static_cast<unsigned char>(lerp(stops[k].g, stops[k + 1].g)),
                static_cast<unsigned char>(lerp(stops[k].b, stops[k + 1].b))
            );
        }
    }
    return sf::Color(253, 231, 37);
}

int main(){
    sf::RenderWindow window(sf::VideoMode({640, 640}), "flu.id");

    std::vector<float> density((N + 2) * (N + 2), 0.0f);
    std::vector<float> densityPrev((N + 2) * (N + 2), 0.0f);
    std::vector<float> vx((N + 2) * (N + 2), 0.0f);
    std::vector<float> vy((N + 2) * (N + 2), 0.0f);
    std::vector<float> p((N + 2) * (N + 2), 0.0f);
    std::vector<float> div((N + 2) * (N + 2), 0.0f);
    std::vector<float> vx0((N + 2) * (N + 2), 0.0f);
    std::vector<float> vy0((N + 2) * (N + 2), 0.0f);

    sf::Image lutImage({2048, 1}, sf::Color::Black);
    for(int i = 0; i < 2048; ++i){
        float d = (i / 2048.0f) * 5.0f;
        float dVis = 1.0f - std::exp(-d * exposure);
        lutImage.setPixel({(unsigned)i, 0}, viridis(dVis));
    }
    sf::Texture colormap;
    if(!colormap.loadFromImage(lutImage)){
        std::cerr << "Failed to load colormap\n";
    }
    colormap.setSmooth(true);
    sf::VertexArray vertices(sf::PrimitiveType::Triangles, N * N * 6);

    float prevMouseX;
    float prevMouseY;
    bool firstMouse = true;

    sf::Clock clock;
    float accumulator = 0.f;
    const float simDt = 1.f / 120.f;
    const float minSpeed = 30.f;
    const float densityDiss = 1.3f;
    const float forceScale = 6.f;
    const float maxForce = 18.f;

    while(window.isOpen()){
        float frameDt = clock.restart().asSeconds();
        frameDt = std::min(frameDt, 0.033f);
        accumulator += frameDt;

        while(const std::optional event = window.pollEvent()){
            if(event->is<sf::Event::Closed>()){
                window.close();
            }
            
            if(const auto* mouseMoved = event->getIf<sf::Event::MouseMoved>()){
                float mouseX = mouseMoved->position.x;
                float mouseY = mouseMoved->position.y;
                
                if(firstMouse){
                    prevMouseX = mouseX;
                    prevMouseY = mouseY;
                    firstMouse = false;
                }

                float dx = mouseX - prevMouseX, dy = mouseY - prevMouseY;
                float speed = std::hypot(dx, dy) / std::max(frameDt, 1e-4f);
                if(speed >= minSpeed){
                    float fx = std::clamp((dx / CELL) * forceScale, -maxForce, maxForce);
                    float fy = std::clamp((dy / CELL) * forceScale, -maxForce, maxForce);

                    int steps = std::max(1, (int)std::max(std::abs(dx), std::abs(dy)));
                    for(int s = 0; s <= steps; ++s){
                        float t = (steps == 0) ? 0.0f : (float)s / steps;

                        float xPos = prevMouseX + t * dx;
                        float yPos = prevMouseY + t * dy;
                        
                        int x = xPos / CELL, y = yPos / CELL;
                        if(x >= 1 && x <= N && y >= 1 && y <= N){
                            for(int i = -2; i <= 2; ++i){
                                for(int j = -2; j <= 2; ++j){
                                    int nx = x + i, ny = y + j;
                                    if(nx >= 1 && nx <= N && ny >= 1 && ny <= N){
                                        float dist2 = i * i + j * j;
                                        float falloff = exp(-dist2 * 0.5f);
                                        density[ix(nx, ny)] += 0.03f * falloff;

                                        vx[ix(nx, ny)] += fx * falloff;
                                        vy[ix(nx, ny)] += fy * falloff;
                                    }
                                }
                            }
                        }
                    }
                }

                prevMouseX = mouseX, prevMouseY = mouseY;
            }
        }

        window.clear(sf::Color::Black);
        
        float diff = 0.000001f, visc = 0.000001f;
        
        const int maxSubsteps = 3;
        int steps = 0;

        while(accumulator >= simDt && steps < maxSubsteps){
            velStep(N, vx, vy, vx0, vy0, p, div, visc, simDt);
            densStep(N, density, densityPrev, vx, vy, diff, simDt);
            #pragma omp parallel for collapse(2)
            for(int i = 1; i <= N; ++i){
                for(int j = 1; j <= N; ++j){
                    density[ix(i, j)] *= std::exp(-densityDiss * simDt);
                }
            }
            
            accumulator -= simDt;
            ++steps;
        }
        
        accumulator = std::min(accumulator, simDt * maxSubsteps);

        #pragma omp parallel for collapse(2)
        for(int i = 1; i <= N; ++i){
            for(int j = 1; j <= N; ++j){
                int vertexIndex = (((i - 1) * N) + (j - 1)) * 6;
                float x0 = (i - 1) * CELL;
                float y0 = (j - 1) * CELL;
                float x1 = (i) * CELL;
                float y1 = (j) * CELL;
                
                float d00 = std::max(0.0f, density[ix(i, j)]);
                float d10 = std::max(0.0f, density[ix(i + 1, j)]);
                float d01 = std::max(0.0f, density[ix(i, j + 1)]);
                float d11 = std::max(0.0f, density[ix(i + 1, j + 1)]);
                
                auto getTexX = [&](float d){
                    float dVis = 1.0f - std::exp(-d * exposure);
                    return std::clamp(dVis * 2047.0f, 0.0f, 2047.0f);
                };

                float tx00 = getTexX(d00);
                float tx10 = getTexX(d10);
                float tx01 = getTexX(d01);
                float tx11 = getTexX(d11);
                
                
                vertices[vertexIndex+0].position = {x0, y0};
                vertices[vertexIndex+0].color = sf::Color::White;
                vertices[vertexIndex+0].texCoords = {tx00, 0.f};

                vertices[vertexIndex+1].position = {x1, y0};
                vertices[vertexIndex+1].color = sf::Color::White;
                vertices[vertexIndex+1].texCoords = {tx10, 0.f};

                vertices[vertexIndex+2].position = {x0, y1};
                vertices[vertexIndex+2].color = sf::Color::White;
                vertices[vertexIndex+2].texCoords = {tx01, 0.f};

                vertices[vertexIndex+3].position = {x0, y1};
                vertices[vertexIndex+3].color = sf::Color::White;
                vertices[vertexIndex+3].texCoords = {tx01, 0.f};

                vertices[vertexIndex+4].position = {x1, y0};
                vertices[vertexIndex+4].color = sf::Color::White;
                vertices[vertexIndex+4].texCoords = {tx10, 0.f};

                vertices[vertexIndex+5].position = {x1, y1};
                vertices[vertexIndex+5].color = sf::Color::White;
                vertices[vertexIndex+5].texCoords = {tx11, 0.f};
            }
        }
        window.draw(vertices, &colormap);
        window.display();
    }
}
