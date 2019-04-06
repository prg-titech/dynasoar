#include <SDL2/SDL.h>

#include "configuration.h"
#include "rendering.h"


// Constants for rendering.
static const int kWindowWidth = 500;
static const int kWindowHeight = 500;

// SDL rendering variables.
SDL_Window* window = nullptr;
SDL_Renderer* renderer = nullptr;

static void render_spring(SDL_Renderer* renderer, const SpringInfo& spring) {
  float force_ratio = spring.force / spring.max_force;
  int r = 0; int g = 0; int b = 0;

  if (force_ratio <= 1.0) {
    r = 255*force_ratio;
  } else {
    b = 255;
  }

  SDL_SetRenderDrawColor(renderer, r, g, b, SDL_ALPHA_OPAQUE);
  SDL_RenderDrawLine(renderer,
                     spring.p1_x*kWindowWidth, spring.p1_y*kWindowWidth,
                     spring.p2_x*kWindowWidth, spring.p2_y*kWindowWidth);

  // Draw endpoints
  SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);

  SDL_Rect rect;
  rect.w = rect.h = 5;
  rect.x = spring.p1_x*kWindowWidth - 2;
  rect.y = spring.p1_y*kWindowHeight - 2;
  SDL_RenderDrawRect(renderer, &rect);

  rect.w = rect.h = 5;
  rect.x = spring.p2_x*kWindowWidth - 2;
  rect.y = spring.p2_y*kWindowHeight - 2;
  SDL_RenderDrawRect(renderer, &rect);
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

