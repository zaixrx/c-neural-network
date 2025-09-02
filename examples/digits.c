#include "nn.h"
#define LOAD_MNIST_IMPLEMENTAION
#include "nn_data_loader.h"
#include <raylib.h>

#define BASE 28
#define SCALE 20
#define SIZE BASE * SCALE
static uint8_t image[SIZE][SIZE];

static int max(int x, int y) { return x >= y ? x : y; }
static int min(int x, int y) { return x < y ? x : y; }

void ApplyBrush(int x, int y, int brush_size) {
	for (int i = x; i <= min(x+brush_size, SIZE-1); ++i) {
		for (int j = y; j <= min(y+brush_size, SIZE-1); ++j) {
			image[i][j] = 0xFF;
		}
	}
}

#define WIN_W (SIZE)
#define WIN_H (SIZE)
#define FPS 60
int main(void) {
	Network net = {0};
	assert(network_import(&net, "nn.data") == NN_CODE_SUCCESS);
	DataEntry *set = load_test_set("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte", 1e4);

	InitWindow(WIN_W, WIN_H, "draw a digit");
    	SetTargetFPS(FPS);

	int brush_size = 15;
    	Vector2 prev_mouse = (Vector2){ -1, -1 };

	bool predict_mode = false;

    	while (!WindowShouldClose()) {
		Vector2 mouse = GetMousePosition();

		if (IsKeyDown(KEY_SPACE)) {
			memset(image, 0, SIZE * SIZE);
		}

		if (IsKeyPressed(KEY_ENTER)) {
			predict_mode = !predict_mode;
		}

    	    	if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
			int i = (int)mouse.x;
    	    	    	int j = (int)mouse.y;

    	    	    	if (i >= 0 && i < SIZE && j >= 0 && j < SIZE) {
    	    	    		if (prev_mouse.x >= 0 && prev_mouse.y >= 0) {
    	    	    	    	    	int x0 = (int)prev_mouse.x;
    	    	    	    	    	int y0 = (int)prev_mouse.y;

					if (x0 >= SIZE) x0 = SIZE-1;
					if (y0 >= SIZE) y0 = SIZE-1;

    	    	    	    	    	int dx = abs(i - x0), sx = x0 < i ? 1 : -1;
                    			int dy = -abs(j - y0), sy = y0 < j ? 1 : -1;
					int err = dx + dy;

					for(;;) {
			    			ApplyBrush(x0, y0, brush_size);

    	    	    	    	    	    	if (x0 == i && y0 == j) break;
    	    	    	    	    	    	int e2 = 2 * err;
    	    	    	    	    	    	if (e2 >= dy) { err += dy; x0 += sx; }
    	    	    	    	    	    	if (dx >= e2) { err += dx; y0 += sy; }
    	    	    	    	    	}
    	    	    	    	} else {
			    		ApplyBrush(i, j, brush_size);
    	    	    	    	}
    	    	    	}
    	    	} else {
    	    	    	prev_mouse.x = -1;
    	    	    	prev_mouse.y = -1;
    	    	}
    	    	prev_mouse = mouse;

    	    	BeginDrawing();
    	    	ClearBackground(BLACK);
		if (predict_mode) {
			vec_t input = vec_new(BASE*BASE);
			for (int x = 0; x < BASE; ++x) {
				for (int y = 0; y < BASE; ++y) {
					float average = 0;
					for (int bx = 0; bx < SCALE; ++bx) {
						for (int by = 0; by < SCALE; ++by) {
							average += image[x*SCALE+bx][y*SCALE+by];
						}
					}
					input[x*BASE+y] = (average /= SCALE*SCALE*255.0F);
					DrawRectangle(
						(x+.5)*SCALE, (y+.5)*SCALE,
						SCALE, SCALE,
						(Color){255, 255, 255, input[x*BASE+y]*255}
					);
				}
			}
			double *out = network_feedforward(&net, input);
			int max = 0; for (int i = 1; i < arrlen(out); ++i) if (out[i] > out[max]) max = i;
			printf("expected %d\n", max);
			vec_destroy(out);
		} else {
			for (int x = 0; x < SIZE; ++x) {
				for (int y = 0; y < SIZE; ++y) {
					if (!image[x][y]) continue;
					DrawPixel(x, y, (Color){255, 255, 255, image[x][y]});
				}
			}
		}
    	    	EndDrawing();
    	}

    	CloseWindow();
    	return 0;
}
