#include <SDL2/SDL.h>

#include "example/nbody/soa/configuration.h"
#include "example/nbody/soa/rendering.h"

namespace nbody {

// Constants for rendering.
static const int kWindowWidth = 1000;
static const int kWindowHeight = 1000;
static const int kMaxRect = 20;

// SDL rendering variables.
SDL_Window* window = nullptr;
SDL_Renderer* renderer = nullptr;

static void render_rect(SDL_Renderer* renderer, float x, float y, float mass) {
  SDL_Rect rect;
  rect.w = rect.h = mass / kMaxMass * kMaxRect;
  rect.x = (x/2 + 0.5) * kWindowWidth - rect.w/2;
  rect.y = (y/2 + 0.5) * kWindowHeight - rect.h/2;
  SDL_RenderDrawRect(renderer, &rect);
}


// Render simulation. Return value indicates if similation should continue.
void draw(float* host_Body_pos_x, float* host_Body_pos_y,
          float* host_Body_mass) {
  // Clear scene.
  SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);
  SDL_RenderClear(renderer);
  SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);

  // Draw all bodies.
  for (int i = 0; i < kNumBodies; ++i) {
    render_rect(renderer,
                host_Body_pos_x[i],
                host_Body_pos_y[i],
                host_Body_mass[i]);
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

}  // namespace nbody
