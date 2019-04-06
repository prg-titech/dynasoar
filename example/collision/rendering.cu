#include <SDL2/SDL.h>

#include "configuration.h"
#include "rendering.h"


// Constants for rendering.
static const int kWindowWidth = 500;
static const int kWindowHeight = 500;

// SDL rendering variables.
SDL_Window* window = nullptr;
SDL_Renderer* renderer = nullptr;

static void render_rect(SDL_Renderer* renderer, float x, float y, float mass,
                        float max_mass) {
  SDL_Rect rect;
  rect.w = rect.h = 10;
  rect.x = (x/2 + 0.5) * kWindowWidth - rect.w/2;
  rect.y = (y/2 + 0.5) * kWindowHeight - rect.h/2;

  int c = (mass/max_mass)*255 + 40;
  c = c > 255 ? 255 : c;
  c = 255 - c;

  SDL_SetRenderDrawColor(renderer, c, c, c, SDL_ALPHA_OPAQUE);
  SDL_RenderFillRect(renderer, &rect);
}


// Render simulation. Return value indicates if similation should continue.
void draw(float* host_Body_pos_x, float* host_Body_pos_y,
          float* host_Body_mass, int num_bodies) {
  // Clear scene.
  SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);
  SDL_RenderClear(renderer);
  SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);

  float max_mass = 0.0f;
  for (int i = 0; i < num_bodies; ++i) {
    max_mass += host_Body_mass[i];
  }
  max_mass /= 3;

  // Draw all bodies.
  for (int i = 0; i < num_bodies; ++i) {
    render_rect(renderer,
                host_Body_pos_x[i],
                host_Body_pos_y[i],
                host_Body_mass[i],
                max_mass);
  }

  SDL_RenderPresent(renderer);

  // Continue until the user closes the window.
  SDL_Event event;
  while (SDL_PollEvent(&event)) {
    if (event.type == SDL_QUIT) {
      exit(1);
    }
  }
}


void init_renderer() {
  // Initialize graphical output.
  if (SDL_Init(SDL_INIT_VIDEO) != 0) {
    printf("Could not initialize SDL!\n");
    exit(1);
  }

  if (SDL_CreateWindowAndRenderer(kWindowWidth, kWindowHeight, 0,
        &window, &renderer) != 0) {
    printf("Could not create window/render!\n");
    exit(1);
  }
}


void close_renderer() {
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();
}
