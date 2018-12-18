#include <SDL2/SDL.h>

#include "configuration.h"
#include "rendering.h"


// Constants for rendering.
static const int kWindowWidth = 1000;
static const int kWindowHeight = 1000;

// SDL rendering variables.
SDL_Window* window = nullptr;
SDL_Renderer* renderer = nullptr;

static void render_spring(SDL_Renderer* renderer, const SpringInfo& spring) {
  SDL_RenderDrawLine(renderer,
                     spring.p1_x*kWindowWidth, spring.p1_y*kWindowWidth,
                     spring.p2_x*kWindowWidth, spring.p2_y*kWindowWidth);
}


// Render simulation. Return value indicates if similation should continue.
void draw(int num_springs, SpringInfo* springs) {
  // Clear scene.
  SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);
  SDL_RenderClear(renderer);
  SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);

  // Draw all bodies.
  for (int i = 0; i < num_springs; ++i) {
    render_spring(renderer, springs[i]);
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

