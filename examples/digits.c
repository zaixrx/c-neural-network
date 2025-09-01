#include "nn.h"
#include <stddef.h>
#include <stdint.h>
#include <raylib.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
	int x;
	int y;
} iVector2;

#define FPS 60
#define BASE_SIZE 28
#define SCALE_SIZE 4
#define IMAGE_SIZE BASE_SIZE * SCALE_SIZE
static uint8_t image[IMAGE_SIZE][IMAGE_SIZE] = {0};

#define PIXEL_SIZE 8
#define WINDOW_SIZE IMAGE_SIZE * PIXEL_SIZE

static inline iVector2 rasterize(Vector2 pos) {
	return (iVector2){
		.x = (int)(pos.x/PIXEL_SIZE),
		.y = (int)(pos.y/PIXEL_SIZE),
	};
}

static inline bool in_bounds(iVector2 pos) {
	return (0 <= pos.x && pos.x < IMAGE_SIZE) && (0 <= pos.y && pos.y < IMAGE_SIZE);
}

static inline bool ivec_equal(iVector2 a, iVector2 b) {
	return a.x == b.x && a.y == b.y;
}

int main(void) {
	iVector2 pos, prev_pos = { -1, -1 };
	InitWindow(WINDOW_SIZE, WINDOW_SIZE, "digits classifier");
	SetTargetFPS(FPS);
	while (!WindowShouldClose()) {
		if (IsKeyDown(KEY_SPACE)) {
			memset(image, 0, sizeof(image));
		}
		pos = rasterize(GetMousePosition());
		if (in_bounds(pos)) {
			if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
				if (prev_pos.x == -1 || ivec_equal(pos, prev_pos)) {
					image[pos.y][pos.x] = 255;
					prev_pos = pos;
				} else {
					int sx = pos.x > prev_pos.x ? 1 : -1;
					int sy = pos.y > prev_pos.y ? 1 : -1;
					iVector2 delta = {
						abs(prev_pos.x - pos.x),
						abs(prev_pos.y - pos.y)
					};
					size_t s = 1;
					while (prev_pos.x != pos.x || prev_pos.y != pos.y) {
						if (delta.x > delta.y) {
							if (delta.y > 0) {
								int q = delta.x/delta.y;
								if (s % q == 0 && prev_pos.y != pos.y) {
									prev_pos.y += sy;
								}
							}
							prev_pos.x += sx;
						} else {
							if (delta.x > 0) {
								int q = delta.y/delta.x;
								if (s % q == 0 && prev_pos.x != pos.x) {
									prev_pos.x += sx;
								}
							}
							prev_pos.y += sy;
						}
						image[prev_pos.y][prev_pos.x] = 255;
						++s;
					}
				}
			} else {
				prev_pos = (iVector2){ -1, -1 };
			}
		}
		BeginDrawing();
			ClearBackground(BLACK);
			vec_t smol = vec_new(BASE_SIZE*BASE_SIZE);
			for (size_t i = 0; i < BASE_SIZE; ++i) {
				for (size_t j = 0; j < BASE_SIZE; ++j) {
					double sum = 0;
					for (size_t bi = 0; bi < SCALE_SIZE; ++bi) {
						for (size_t bj = 0; bj < SCALE_SIZE; ++bj) {
							sum += image[bi+i*SCALE_SIZE][bj+j*SCALE_SIZE];
						}
					}
					smol[i*BASE_SIZE+j] = sum / (SCALE_SIZE*SCALE_SIZE);
				}
			}

			for (size_t i = 0; i < IMAGE_SIZE; ++i) {
				for (size_t j = 0; j < IMAGE_SIZE; ++j) {
					if (image[i][j] == 0) continue;
					DrawRectangle((j+1)*PIXEL_SIZE, (i+1)*PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE, WHITE);
				}
			}

			for (size_t i = 0; i < BASE_SIZE*BASE_SIZE; ++i) {
				if (!smol[i]) continue;
				printf("%f\n", smol[i]);
				Color c = WHITE;
				c.a = smol[i];
				DrawRectangle((i%BASE_SIZE)*PIXEL_SIZE, (i/BASE_SIZE)*PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE, c);
			}
		EndDrawing();
	}
	CloseWindow();
	return 0;
}
