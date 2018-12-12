#include <SDL2/SDL.h>

#include "configuration.h"
#include "rendering.h"

// Constants for rendering.
static const int kCellWidth = 4;

// SDL rendering variables.
SDL_Window* window = nullptr;
SDL_Renderer* renderer = nullptr;

static void render_rect(SDL_Renderer* renderer, int x, int y, CellInfo& info) {
  SDL_Rect rect;
  rect.w = rect.h = kCellWidth;
  rect.x = x*kCellWidth;
  rect.y = y*kCellWidth;

  float sugar_level = static_cast<float>(info.sugar) / kSugarCapacity;
  int sugar_level_int = sugar_level*255;
  if (sugar_level_int > 255) sugar_level_int = 255;

  SDL_SetRenderDrawColor(renderer, sugar_level_int,
                         sugar_level_int, 0, SDL_ALPHA_OPAQUE);
  SDL_RenderFillRect(renderer, &rect);

  if (info.agent_type != 0) {
    SDL_Rect rect;
    rect.w = rect.h = kCellWidth - 2;
    rect.x = x*kCellWidth + 1;
    rect.y = y*kCellWidth + 1;

    if (info.agent_type == 1) {
      // Male agent
      SDL_SetRenderDrawColor(renderer, 0, 0, 255, SDL_ALPHA_OPAQUE);
    } else if (info.agent_type == 2) {
      // Female agent
      SDL_SetRenderDrawColor(renderer, 255, 0, 0, SDL_ALPHA_OPAQUE);
    }

    SDL_RenderFillRect(renderer, &rect);
  }
}


// Render simulation. Return value indicates if similation should continue.
void draw(CellInfo* cell_info) {
  // Clear scene.
  SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);
  SDL_RenderClear(renderer);

  // Draw all bodies.
  for (int i = 0; i < kSize * kSize; ++i) {
    render_rect(renderer, i%kSize, i/kSize, cell_info[i]);
  }

  SDL_RenderPresent(renderer);

  // Continue until the user closes the window.
  SDL_Event event;
  while (SDL_PollEvent(&event)) {
    if (event.type == SDL_QUIT) {
      exit(1);
    }
  }

  SDL_Delay(50);
}


void init_renderer() {
  // Initialize graphical output.
  if (SDL_Init(SDL_INIT_VIDEO) != 0) {
    printf("Could not initialize SDL!\n");
    exit(1);
  }

  if (SDL_CreateWindowAndRenderer(kCellWidth*kSize, kCellWidth*kSize,
        0, &window, &renderer) != 0) {
    printf("Could not create window/render!\n");
    exit(1);
  }
}


void close_renderer() {
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();
}

