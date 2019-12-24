#include <SDL2/SDL.h>

#include "configuration.h"
#include "rendering.h"


// Constants for rendering.
#ifdef PARAM_RENDER_SCALE
static const float kRenderScale = PARAM_RENDER_SCALE;
#else
static const float kRenderScale = 1.0f;
#endif  // PARAM_RENDER_SCALE

static const int kWindowWidth = 500*kRenderScale;
static const int kWindowHeight = 500*kRenderScale;

// SDL rendering variables.
SDL_Window* window = nullptr;
SDL_Renderer* renderer = nullptr;

void draw_body(float x, float y, float mass, float max_mass) {
  SDL_Rect rect;
  rect.w = rect.h = 5*kRenderScale;
  rect.x = (x/2 + 0.5) * kWindowWidth - rect.w/2;
  rect.y = (y/2 + 0.5) * kWindowHeight - rect.h/2;

  int c = 3*(mass/max_mass)*255 + 150;
  c = c > 255 ? 255 : c;
  c = 255 - c;

  SDL_SetRenderDrawColor(renderer, c, c, c, SDL_ALPHA_OPAQUE);
  SDL_RenderFillRect(renderer, &rect);
}


void draw_tree_node(float x1, float y1, float x2, float y2) {
  SDL_Rect rect;
  rect.w = (x2 - x1) * kWindowWidth;
  rect.h = (y2 - y1) * kWindowHeight;
  rect.x = (x1/2 + 0.5) * kWindowWidth;
  rect.y = (y1/2 + 0.5) * kWindowHeight;

  SDL_SetRenderDrawColor(renderer, 0, 0, 255, SDL_ALPHA_OPAQUE);
  SDL_RenderDrawRect(renderer, &rect);
}


void init_frame() {
  // Clear scene.
  SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);
  SDL_RenderClear(renderer);
  SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
}


void show_frame() {
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
