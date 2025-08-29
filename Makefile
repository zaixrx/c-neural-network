main.out: src/main.c src/nn.c
	cc -o main.out src/main.c src/nn.c -I./include -lm -O3
