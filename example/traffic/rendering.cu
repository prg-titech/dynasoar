#include <assert.h>
#include <SDL2/SDL.h>

#include "configuration.h"
#include "rendering.h"


// Constants for rendering.
static const int kWindowWidth = 500;
static const int kWindowHeight = 500;

// SDL rendering variables.
SDL_Window* window = nullptr;
SDL_Renderer* renderer = nullptr;

static void render_rect(SDL_Renderer* renderer, float x, float y,
                        bool occupied) {
  assert(x >= 0 && x <= 1);
  assert(y >= 0 && y <= 1);

  SDL_Rect rect;
  rect.w = rect.h = 3;
  rect.x = x * kWindowWidth - rect.w/2;
  rect.y = y * kWindowHeight - rect.h/2;

  int c;
  if (occupied) {
    c = 0;
  } else {
    c = 200;
  }

  SDL_SetRenderDrawColor(renderer, c, c, c, SDL_ALPHA_OPAQUE);
  SDL_RenderFillRect(renderer, &rect);
}


// Render simulation. Return value indicates if similation should continue.
void draw(float* host_Cell_pos_x, float* host_Cell_pos_y,
          bool* host_Cell_occupied, int num_cells) {
  SDL_Delay(80);

  // Clear scene.
  SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);
  SDL_RenderClear(renderer);
  SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);

  // Draw all cells.
  for (int i = 0; i < num_cells; ++i) {
    render_rect(renderer,
                host_Cell_pos_x[i],
                host_Cell_pos_y[i],
                host_Cell_occupied[i]);
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
