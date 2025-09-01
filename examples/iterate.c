#include "nn.h"
#include <raylib.h>
#include <unistd.h>
#define LOAD_MNIST_IMPLEMENTAION
#include "nn_data_loader.h"

typedef struct {
	int x;
	int y;
} iVector2;

#define FPS 60
#define BASE_SIZE 28
#define PIXEL_SIZE 20
#define WIN_SIZE BASE_SIZE * PIXEL_SIZE

bool Render(Network *net, DataEntry *set) {
	static size_t t = 0;
	while (!WindowShouldClose() && t < arrlen(set)) {
		DataEntry entry = set[t];
		vec_t image = entry.x;
		vec_t result = entry.y;
		++t;
		BeginDrawing();
			ClearBackground(BLACK);
			for (size_t i = 0; i < BASE_SIZE*BASE_SIZE; ++i) {
				if (!image[i]) continue;
				Color c = WHITE; c.a = image[i] * 255;
				DrawRectangle((i%BASE_SIZE)*PIXEL_SIZE, (i/BASE_SIZE)*PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE, c);
			}
			double *out = network_feedforward(net, image);
			size_t predicted = 0; for (size_t i = 1; i < arrlen(out); ++i) if (out[i] > out[predicted]) predicted = i;
			size_t actual = 0; for (size_t i = 1; i < arrlen(result); ++i) if (result[i] > result[actual]) actual = i;
			if (actual != predicted) {
				char text = '0'+predicted;
				DrawText(&text, 0, 0, 50, RED);
				EndDrawing();
				break;
			}
		EndDrawing();
	}
	return t == arrlen(set);
}

int main(void) {
	Network net = {0};
	assert(network_import(&net, "nn.data") == NN_CODE_SUCCESS && "Train your data before");
	DataEntry *test_set = load_test_set("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte", 1e4);
	InitWindow(WIN_SIZE, WIN_SIZE, "digits classifier");
	SetTargetFPS(FPS);
	while(!Render(&net, test_set)) {
		usleep(500000);
	}
	CloseWindow();
	return 0;
}
