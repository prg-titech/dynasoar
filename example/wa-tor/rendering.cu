#include <assert.h>
#include <SDL2/SDL.h>

#include "configuration.h"
#include "rendering.h"


// SDL rendering variables.
SDL_Window* window = nullptr;
SDL_Renderer* renderer = nullptr;


// Render simulation. Return value indicates if similation should continue.
void draw(char* pixels) {
  // Clear scene.
  SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);
  SDL_RenderClear(renderer);
  SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);

  // Draw all cells.
  for (int i = 0; i < kSizeX*kSizeY; ++i) {
    int pos_x = i % kSizeX;
    int pos_y = i / kSizeX;

    if (pixels[i] == 1) {
      SDL_SetRenderDrawColor(renderer, 0, 255, 0, SDL_ALPHA_OPAQUE);
    } else if (pixels[i] == 2) {
      SDL_SetRenderDrawColor(renderer, 255, 0, 0, SDL_ALPHA_OPAQUE);
    } else {
      SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
    }

    SDL_RenderDrawPoint(renderer, pos_x, pos_y);
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

  if (SDL_CreateWindowAndRenderer(kSizeX, kSizeY, 0,
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
