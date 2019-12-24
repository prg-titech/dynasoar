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


static void render_line(SDL_Renderer* renderer, float x1, float y1,
                        float x2, float y2, int c) {
  int px1 = (x1/2 + 0.5) * kWindowWidth;
  int py1 = (y1/2 + 0.5) * kWindowHeight;
  int px2 = (x2/2 + 0.5) * kWindowWidth;
  int py2 = (y2/2 + 0.5) * kWindowHeight;  

  SDL_SetRenderDrawColor(renderer, c, 0, 0, SDL_ALPHA_OPAQUE);
  SDL_RenderDrawLine(renderer, px1, py1, px2, py2);
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


void draw_body(float x, float y, float mass, float max_mass) {
  SDL_Rect rect;
  // rect.w = rect.h = 10;
  rect.w = rect.h = pow(mass/kMaxMass, 0.125) * 15;
  rect.x = (x/2 + 0.5) * kWindowWidth - rect.w/2;
  rect.y = (y/2 + 0.5) * kWindowHeight - rect.h/2;

  int c = (3*mass/max_mass)*255 + 40;
  c = c > 255 ? 255 : c;
  c = 255 - c;

  SDL_SetRenderDrawColor(renderer, c, c, c, SDL_ALPHA_OPAQUE);
  //SDL_RenderFillRect(renderer, &rect);

  int cx = (x/2 + 0.5) * kWindowWidth;
  int cy = (y/2 + 0.5) * kWindowHeight;
  int radius = pow(mass/kMaxMass, 0.125) * 8;

  for (double dy = 1; dy <= radius; dy += 1.0) {
    double dx = floor(sqrt((2.0 * radius * dy) - (dy * dy)));
    SDL_RenderDrawLine(renderer, cx - dx, cy + dy - radius, cx + dx, cy + dy - radius);
    SDL_RenderDrawLine(renderer, cx - dx, cy - dy + radius, cx + dx, cy - dy + radius);
  }
}


void maybe_draw_line(float pos_x, float pos_x2, float pos_y, float pos_y2) {
  float dx = pos_x2 - pos_x;
  float dy = pos_y2 - pos_y;
  float dist_sq = dx*dx + dy*dy;

  if (dist_sq < (15*kMergeThreshold)*(15*kMergeThreshold)) {
    int color = 255 - dist_sq / ((15*kMergeThreshold)*(15*kMergeThreshold)) * 255;
    render_line(renderer, pos_x, pos_y, pos_x2, pos_y2, color);
  }
}

