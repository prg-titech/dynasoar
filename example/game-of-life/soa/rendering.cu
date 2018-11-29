#include <SDL2/SDL.h>

#include "example/game-of-life/soa/configuration.h"
#include "example/game-of-life/soa/rendering.h"

// Constants for rendering.
static const int kCellWidth = 2;

// SDL rendering variables.
SDL_Window* window = nullptr;
SDL_Renderer* renderer = nullptr;

static void render_rect(SDL_Renderer* renderer, int x, int y, char state) {
  SDL_Rect rect;
  rect.w = rect.h = kCellWidth;
  rect.x = x*kCellWidth;
  rect.y = y*kCellWidth;
  SDL_RenderDrawRect(renderer, &rect);
}


// Render simulation. Return value indicates if similation should continue.
void draw(char* host_cells) {
  // Clear scene.
  SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);
  SDL_RenderClear(renderer);
  SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);

  // Draw all bodies.
  for (int i = 0; i < SIZE_X*SIZE_Y; ++i) {
    render_rect(renderer, i%SIZE_X, i/SIZE_X, host_cells[i]);
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

  if (SDL_CreateWindowAndRenderer(kCellWidth*SIZE_X, kCellWidth*SIZE_Y, 0,
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

